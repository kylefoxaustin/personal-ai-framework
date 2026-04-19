#!/usr/bin/env python3
"""
Batch Email Ingestion from Extracted PST Files
Processes extensionless email files with progress tracking
"""
import os
import sys
import email
from email import policy
from pathlib import Path
from datetime import datetime
import requests
import json
import hashlib

from ingest_auth import login_headers
from ingest_failure_log import FailureLog

# Configuration
LLM_SERVER = "http://localhost:8080"
BATCH_SIZE = 50  # Emails per API call
MANIFEST_FILE = Path("knowledge/emails/.pst_ingest_manifest.json")
FAILURE_LOG_PATH = Path("knowledge/emails/.pst_ingest_failures.jsonl")
AUTH_HEADERS = login_headers(LLM_SERVER)
FAILURES = FailureLog(FAILURE_LOG_PATH)

def load_manifest():
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return set(json.load(f))
    return set()

def save_manifest(ingested):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(list(ingested), f)

def parse_email_file(filepath):
    """Parse an RFC822 email file."""
    try:
        with open(filepath, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        parts = []
        
        # Headers
        for header in ['From', 'To', 'Subject', 'Date']:
            val = msg.get(header)
            if val:
                parts.append(f"{header}: {val}")
        
        parts.append("")
        
        # Body
        body = msg.get_body(preferencelist=('plain', 'html'))
        if body:
            content = body.get_content()
            if isinstance(content, str):
                # Strip excessive whitespace and HTML tags (basic)
                content = content.strip()
                if len(content) > 50:  # Skip near-empty emails
                    parts.append(content)
        
        text = '\n'.join(parts)
        return text if len(text) > 100 else None  # Skip very short emails
        
    except Exception as e:
        return None

def _post_batch(documents):
    """Single HTTP POST. Returns chunks_added on 2xx, raises otherwise."""
    response = requests.post(
        f"{LLM_SERVER}/ingest/batch",
        json={'documents': documents},
        headers=AUTH_HEADERS,
        timeout=120
    )
    response.raise_for_status()
    return response.json().get('chunks_added', 0)


def ingest_batch(emails_data):
    """Send a batch of emails, bisecting on failure so one bad email
    doesn't take the whole batch down. Returns total chunks_added."""
    items = [(fp, {'content': c, 'metadata': m}) for fp, c, m in emails_data]
    return _ingest_items(items)


def _ingest_items(items):
    if not items:
        return 0
    try:
        return _post_batch([doc for _, doc in items])
    except Exception as e:
        if len(items) == 1:
            fp = items[0][0]
            print(f"\n    ⚠️ Dropping email {fp}: {e}")
            FAILURES.record(str(fp), str(e))
            return 0
        mid = len(items) // 2
        return _ingest_items(items[:mid]) + _ingest_items(items[mid:])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sent-only', action='store_true', help='Only process Sent Items')
    parser.add_argument('--limit', type=int, help='Limit number of emails to process')
    parser.add_argument('--force', action='store_true', help='Ignore manifest, re-ingest all')
    parser.add_argument('--replay', metavar='LOG', help='Re-attempt emails recorded in a failure log')
    args = parser.parse_args()

    extracted_dir = Path("knowledge/emails/extracted")

    # Replay mode pulls paths from the failure log instead of scanning.
    if args.replay:
        log = FailureLog(args.replay)
        email_files = [Path(s) for s in log.read_sources() if Path(s).exists()]
        if not email_files:
            print(f"⚠️ No replayable emails in {args.replay}")
            return
        print(f"🔁 Replaying {len(email_files)} email(s) from {args.replay}")
        log.archive()
    elif args.sent_only:
        email_files = list(extracted_dir.rglob("*"))
        email_files = [f for f in email_files if f.is_file() and "Sent" in str(f)]
    else:
        email_files = list(extracted_dir.rglob("*"))
        email_files = [f for f in email_files if f.is_file()
                      and ("Inbox" in str(f) or "Sent" in str(f))
                      and "Calendar" not in str(f)
                      and "Contacts" not in str(f)
                      and "Tasks" not in str(f)]

    # Filter to only files (not dirs) with no extension (actual emails)
    email_files = [f for f in email_files if f.is_file() and f.suffix == '']

    if args.limit:
        email_files = email_files[:args.limit]
    
    # Load manifest
    ingested = set() if args.force else load_manifest()
    
    # Filter already ingested
    to_process = [f for f in email_files if str(f) not in ingested]
    
    print("=" * 60)
    print("📧 PST Email Batch Ingestion")
    print("=" * 60)
    print(f"Total email files found: {len(email_files)}")
    print(f"Already ingested: {len(ingested)}")
    print(f"To process: {len(to_process)}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)
    
    if not to_process:
        print("✅ Nothing to ingest!")
        return
    
    # Process in batches
    batch = []
    total_chunks = 0
    processed = 0
    skipped = 0
    errors = 0
    start_time = datetime.now()
    
    for i, filepath in enumerate(to_process):
        # Progress bar
        pct = ((i + 1) / len(to_process)) * 100
        bar_len = 40
        filled = int(bar_len * (i + 1) / len(to_process))
        bar = '█' * filled + '░' * (bar_len - filled)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
        
        print(f"\r[{bar}] {pct:5.1f}% | {i+1}/{len(to_process)} | {rate:.1f}/s | ETA: {eta:.0f}s | Chunks: {total_chunks}", end='', flush=True)
        
        # Parse email
        content = parse_email_file(filepath)
        if not content:
            skipped += 1
            continue
        
        # Get folder context for metadata
        rel_path = filepath.relative_to(extracted_dir)
        parts = rel_path.parts
        metadata = {
            'source': 'pst_email',
            'folder': '/'.join(parts[:-1]) if len(parts) > 1 else 'root',
            'file': filepath.name
        }
        
        batch.append((str(filepath), content, metadata))
        
        # Send batch when full
        if len(batch) >= BATCH_SIZE:
            chunks = ingest_batch(batch)
            total_chunks += chunks
            for fp, _, _ in batch:
                ingested.add(fp)
            processed += len(batch)
            batch = []
            
            # Save manifest periodically
            if processed % 500 == 0:
                save_manifest(ingested)
    
    # Final batch
    if batch:
        chunks = ingest_batch(batch)
        total_chunks += chunks
        for fp, _, _ in batch:
            ingested.add(fp)
        processed += len(batch)
    
    # Save final manifest
    save_manifest(ingested)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print()
    print()
    print("=" * 60)
    print("📊 Ingestion Complete!")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (empty/short): {skipped}")
    print(f"Chunks added: {total_chunks}")
    print(f"Time: {elapsed:.1f}s ({processed/elapsed:.1f} emails/s)")
    print(f"Manifest saved: {MANIFEST_FILE}")

if __name__ == '__main__':
    main()
