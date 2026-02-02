#!/usr/bin/env python3
"""
Knowledge Base Ingestion Script
Recursively scans folders, tracks changes, and ingests documents into RAG pipeline.
"""
import os
import sys
import json
import hashlib
import argparse
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import yaml
import requests
import email
import mailbox
import fitz  # PyMuPDF
from email import policy


class IngestManifest:
    """Tracks ingested files to avoid duplicates."""

    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path)
        self.data: Dict[str, dict] = {}
        self.load()

    def load(self):
        if self.path.exists():
            with open(self.path, 'r') as f:
                self.data = json.load(f)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def get_hash(self, filepath: Path) -> str:
        """Compute MD5 hash of file contents."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def needs_update(self, filepath: Path) -> bool:
        """Check if file is new or changed."""
        key = str(filepath)
        current_hash = self.get_hash(filepath)

        if key not in self.data:
            return True
        if self.data[key].get('hash') != current_hash:
            return True
        return False

    def mark_ingested(self, filepath: Path, chunks: int):
        """Record that a file has been ingested."""
        self.data[str(filepath)] = {
            'hash': self.get_hash(filepath),
            'ingested_at': datetime.now().isoformat(),
            'chunks': chunks
        }


class DocumentParser:
    """Parse different file types into text content."""

    @staticmethod
    def parse_txt(filepath: Path) -> str:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def parse_md(filepath: Path) -> str:
        return DocumentParser.parse_txt(filepath)

    @staticmethod
    def parse_json(filepath: Path) -> str:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Handle common JSON structures
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            # Look for content fields
            for key in ['content', 'text', 'body', 'message']:
                if key in data:
                    return str(data[key])
            return json.dumps(data, indent=2)
        if isinstance(data, list):
            return '\n\n'.join(str(item) for item in data)
        return str(data)

    @staticmethod
    def parse_eml(filepath: Path) -> str:
        with open(filepath, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        parts = []

        # Headers
        if msg['From']:
            parts.append(f"From: {msg['From']}")
        if msg['To']:
            parts.append(f"To: {msg['To']}")
        if msg['Subject']:
            parts.append(f"Subject: {msg['Subject']}")
        if msg['Date']:
            parts.append(f"Date: {msg['Date']}")

        parts.append("")  # Blank line

        # Body
        body = msg.get_body(preferencelist=('plain', 'html'))
        if body:
            content = body.get_content()
            if isinstance(content, str):
                parts.append(content)

        return '\n'.join(parts)

    @staticmethod
    def parse_pdf(filepath: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        doc = fitz.open(filepath)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)

    @staticmethod
    def parse_mbox(filepath: Path) -> str:
        """Extract all emails from mbox file."""
        mbox = mailbox.mbox(filepath)
        emails = []
        for i, message in enumerate(mbox):
            parts = []
            if message.get("From", ""):
                parts.append(f'From: {message.get("From", "")}')
            if message.get("To", ""):
                parts.append(f'To: {message.get("To", "")}')
            if message.get("Subject", ""):
                parts.append(f'Subject: {message.get("Subject", "")}')
            if message.get("Date", ""):
                parts.append(f'Date: {message.get("Date", "")}')
            body = message.get_payload(decode=True)
            if body:
                try:
                    body = body.decode("utf-8", errors="ignore")
                    parts.append(body)
                except:
                    pass
            if parts:
                emails.append("\n".join(parts))
            if i >= 10000:  # Safety limit
                break
        mbox.close()
        return "\n\n---EMAIL---\n\n".join(emails)

    @classmethod
    def parse(cls, filepath: Path) -> Optional[str]:
        """Parse a file based on extension."""
        ext = filepath.suffix.lower()

        parsers = {
            '.txt': cls.parse_txt,
            '.md': cls.parse_md,
            '.json': cls.parse_json,
            '.eml': cls.parse_eml,
            '.pdf': cls.parse_pdf,
            '.mbox': cls.parse_mbox,
        }

        parser = parsers.get(ext)
        if parser:
            try:
                return parser(filepath)
            except Exception as e:
                print(f"  ⚠️ Error parsing {filepath}: {e}")
                return None
        return None


class KnowledgeIngester:
    """Main ingestion orchestrator."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.root = Path(self.config['knowledge_root']).resolve()
        self.manifest = IngestManifest(self.config['manifest_file'])
        self.server_url = self.config['llm_server_url']

        # Stats
        self.stats = {
            'files_scanned': 0,
            'files_skipped': 0,
            'files_ingested': 0,
            'chunks_added': 0,
            'errors': 0
        }

    def should_exclude(self, path: Path) -> bool:
        """Check if path matches exclusion patterns."""
        name = path.name
        for pattern in self.config['exclude_patterns']:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def get_metadata(self, filepath: Path) -> dict:
        """Extract metadata from file path."""
        rel_path = filepath.relative_to(self.root)
        parts = rel_path.parts

        metadata = {
            'source_file': str(rel_path),
            'file_type': filepath.suffix.lower(),
            'ingested_at': datetime.now().isoformat()
        }

        # Use folder structure for categorization
        if len(parts) > 1:
            metadata['category'] = parts[0]
        if len(parts) > 2:
            metadata['subcategory'] = parts[1]

        return metadata

    def ingest_file(self, filepath: Path) -> bool:
        """Ingest a single file."""
        content = DocumentParser.parse(filepath)
        if not content:
            return False

        metadata = self.get_metadata(filepath)

        try:
            response = requests.post(
                f"{self.server_url}/ingest",
                json={
                    'content': content,
                    'metadata': metadata
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            chunks = result.get('chunks_added', 1)
            self.manifest.mark_ingested(filepath, chunks)
            self.stats['chunks_added'] += chunks
            return True

        except Exception as e:
            print(f"  ❌ Ingestion failed: {e}")
            self.stats['errors'] += 1
            return False

    def scan_and_ingest(self, force: bool = False):
        """Scan knowledge folder and ingest new/changed files."""
        extensions = set(self.config['include_extensions'])

        print(f"📁 Scanning: {self.root}")
        print(f"📋 Extensions: {', '.join(extensions)}")
        print("-" * 50)

        for root, dirs, files in os.walk(self.root):
            root_path = Path(root)

            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]

            for filename in files:
                filepath = root_path / filename

                if self.should_exclude(filepath):
                    continue

                if filepath.suffix.lower() not in extensions:
                    continue

                self.stats['files_scanned'] += 1
                rel_path = filepath.relative_to(self.root)

                # Check if needs update
                if not force and not self.manifest.needs_update(filepath):
                    self.stats['files_skipped'] += 1
                    continue

                print(f"📄 Ingesting: {rel_path}")

                if self.ingest_file(filepath):
                    self.stats['files_ingested'] += 1
                    print(f"   ✅ Done")

        # Save manifest
        self.manifest.save()

        # Print summary
        print("-" * 50)
        print("📊 Ingestion Complete!")
        print(f"   Files scanned:  {self.stats['files_scanned']}")
        print(f"   Files ingested: {self.stats['files_ingested']}")
        print(f"   Files skipped:  {self.stats['files_skipped']} (unchanged)")
        print(f"   Chunks added:   {self.stats['chunks_added']}")
        if self.stats['errors']:
            print(f"   ⚠️ Errors:      {self.stats['errors']}")


def main():
    parser = argparse.ArgumentParser(description='Ingest knowledge base into RAG pipeline')
    parser.add_argument('--config', '-c', default='pipeline/ingest_config.yaml',
                        help='Path to config file')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-ingestion of all files')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be ingested without doing it')

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    ingester = KnowledgeIngester(args.config)

    if args.dry_run:
        print("🔍 DRY RUN - no changes will be made")
        # TODO: implement dry run mode

    ingester.scan_and_ingest(force=args.force)


if __name__ == '__main__':
    main()
