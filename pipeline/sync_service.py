#!/usr/bin/env python3
"""
Smart Sync Service for Personal AI Framework
Watches for file changes with debouncing and handles proper chunk cleanup.
"""
import os
import sys
import time
import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import requests
import yaml

# Configuration
DEFAULT_CONFIG = {
    'knowledge_root': './knowledge',
    'debounce_seconds': 300,  # 5 minutes
    'llm_server_url': 'http://localhost:8080',
    'include_extensions': ['.txt', '.md', '.json', '.eml', '.pdf', '.mbox'],
    'exclude_patterns': ['__pycache__', '.git', '.DS_Store', '*.pyc', '.ingest_manifest.json', 'extracted'],
}


class SyncState:
    """Tracks file states and pending changes."""
    
    def __init__(self, state_file: str = './knowledge/.sync_state.json'):
        self.state_file = Path(state_file)
        self.file_hashes: Dict[str, str] = {}
        self.pending_changes: Dict[str, datetime] = {}
        self.load()
    
    def load(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                self.file_hashes = data.get('file_hashes', {})
    
    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump({'file_hashes': self.file_hashes}, f, indent=2)
    
    def get_hash(self, filepath: Path) -> str:
        """Compute MD5 hash of file."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def has_changed(self, filepath: Path) -> bool:
        """Check if file is new or modified."""
        key = str(filepath)
        try:
            current_hash = self.get_hash(filepath)
        except:
            return False
        
        if key not in self.file_hashes:
            return True
        return self.file_hashes[key] != current_hash
    
    def mark_synced(self, filepath: Path):
        """Record file as synced."""
        key = str(filepath)
        try:
            self.file_hashes[key] = self.get_hash(filepath)
        except:
            pass
    
    def mark_deleted(self, filepath: Path):
        """Remove file from tracking."""
        key = str(filepath)
        self.file_hashes.pop(key, None)
    
    def add_pending(self, filepath: Path):
        """Add file to pending changes with timestamp."""
        self.pending_changes[str(filepath)] = datetime.now()
    
    def remove_pending(self, filepath: Path):
        """Remove file from pending."""
        self.pending_changes.pop(str(filepath), None)
    
    def get_ready_changes(self, debounce_seconds: int) -> Set[str]:
        """Get files that haven't changed in debounce_seconds."""
        ready = set()
        cutoff = datetime.now() - timedelta(seconds=debounce_seconds)
        
        for filepath, timestamp in list(self.pending_changes.items()):
            if timestamp < cutoff:
                ready.add(filepath)
        
        return ready


class SyncHandler(FileSystemEventHandler):
    """Handles file system events."""
    
    def __init__(self, state: SyncState, config: dict):
        self.state = state
        self.config = config
        self.extensions = set(config['include_extensions'])
        self.exclude = config['exclude_patterns']
    
    def should_process(self, path: str) -> bool:
        """Check if file should be processed."""
        p = Path(path)
        
        # Check extension
        if p.suffix.lower() not in self.extensions:
            return False
        
        # Check exclusions
        for pattern in self.exclude:
            if pattern in path:
                return False
        
        return True
    
    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        if self.should_process(event.src_path):
            print(f"📄 New file detected: {event.src_path}")
            self.state.add_pending(Path(event.src_path))
    
    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        if self.should_process(event.src_path):
            print(f"📝 File modified: {event.src_path}")
            self.state.add_pending(Path(event.src_path))
    
    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return
        if self.should_process(event.src_path):
            print(f"🗑️  File deleted: {event.src_path}")
            # Mark for immediate deletion from index
            self.state.pending_changes[event.src_path] = datetime.min


class SmartSync:
    """Main sync orchestrator."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = DEFAULT_CONFIG.copy()
        
        # Load config if exists
        if Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                if user_config and 'knowledge' in user_config:
                    for key in ['root_folder', 'include_extensions', 'exclude_patterns']:
                        if key in user_config['knowledge']:
                            mapped_key = 'knowledge_root' if key == 'root_folder' else key
                            self.config[mapped_key] = user_config['knowledge'][key]
        
        self.state = SyncState()
        self.observer = None
        self.running = False
    
    def sync_file(self, filepath: str, deleted: bool = False) -> dict:
        """Sync a single file - delete old chunks, re-ingest if not deleted."""
        result = {'file': filepath, 'deleted_chunks': 0, 'added_chunks': 0, 'status': 'ok'}
        
        try:
            # Always try to delete old chunks first
            rel_path = str(Path(filepath).relative_to(self.config['knowledge_root']))
            
            delete_resp = requests.post(
                f"{self.config['llm_server_url']}/sync/delete",
                json={'source_file': rel_path},
                timeout=30
            )
            if delete_resp.ok:
                result['deleted_chunks'] = delete_resp.json().get('deleted_count', 0)
            
            # If file still exists, re-ingest it
            if not deleted and Path(filepath).exists():
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                # Try to decode as text
                try:
                    text_content = content.decode('utf-8')
                except:
                    text_content = content.decode('latin-1', errors='ignore')
                
                ingest_resp = requests.post(
                    f"{self.config['llm_server_url']}/ingest",
                    json={
                        'content': text_content,
                        'metadata': {'source_file': rel_path}
                    },
                    timeout=60
                )
                if ingest_resp.ok:
                    result['added_chunks'] = ingest_resp.json().get('chunks_added', 0)
                
                self.state.mark_synced(Path(filepath))
            else:
                self.state.mark_deleted(Path(filepath))
            
        except Exception as e:
            result['status'] = f'error: {e}'
        
        return result
    
    def sync_all_pending(self) -> dict:
        """Process all pending changes that have passed the debounce period."""
        ready = self.state.get_ready_changes(self.config['debounce_seconds'])
        
        if not ready:
            return {'synced': 0, 'results': []}
        
        print(f"\n🔄 Syncing {len(ready)} files...")
        results = []
        
        for filepath in ready:
            deleted = self.state.pending_changes.get(filepath) == datetime.min
            result = self.sync_file(filepath, deleted=deleted)
            results.append(result)
            
            if result['status'] == 'ok':
                print(f"  ✅ {Path(filepath).name}: -{result['deleted_chunks']} +{result['added_chunks']}")
            else:
                print(f"  ❌ {Path(filepath).name}: {result['status']}")
            
            self.state.remove_pending(Path(filepath))
        
        self.state.save()
        
        return {'synced': len(results), 'results': results}
    
    def sync_now(self) -> dict:
        """Force immediate sync of all pending changes."""
        # Move all pending to "ready" by setting old timestamps
        for filepath in list(self.state.pending_changes.keys()):
            if self.state.pending_changes[filepath] != datetime.min:
                self.state.pending_changes[filepath] = datetime.min + timedelta(days=1)
        
        return self.sync_all_pending()
    
    def full_sync(self) -> dict:
        """Full sync - scan all files and sync changes."""
        print("🔍 Scanning for changes...")
        
        root = Path(self.config['knowledge_root'])
        extensions = set(self.config['include_extensions'])
        
        files_checked = 0
        changes_found = 0
        
        for filepath in root.rglob('*'):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in extensions:
                continue
            
            # Check exclusions
            skip = False
            for pattern in self.config['exclude_patterns']:
                if pattern in str(filepath):
                    skip = True
                    break
            if skip:
                continue
            
            files_checked += 1
            
            if self.state.has_changed(filepath):
                self.state.add_pending(filepath)
                changes_found += 1
        
        print(f"📊 Checked {files_checked} files, found {changes_found} changes")
        
        if changes_found > 0:
            # Force immediate sync for full sync
            return self.sync_now()
        
        return {'synced': 0, 'files_checked': files_checked}
    
    def start_watching(self):
        """Start the file watcher."""
        self.running = True
        handler = SyncHandler(self.state, self.config)
        
        self.observer = Observer()
        self.observer.schedule(handler, self.config['knowledge_root'], recursive=True)
        self.observer.start()
        
        print(f"👁️  Watching {self.config['knowledge_root']} for changes...")
        print(f"⏱️  Debounce: {self.config['debounce_seconds']} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                time.sleep(10)  # Check for ready changes every 10 seconds
                self.sync_all_pending()
        except KeyboardInterrupt:
            self.stop_watching()
    
    def stop_watching(self):
        """Stop the file watcher."""
        self.running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()
        print("\n👋 Watcher stopped")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Smart Sync for Personal AI')
    parser.add_argument('command', choices=['watch', 'sync', 'full-sync', 'status'],
                        help='Command to run')
    parser.add_argument('--debounce', type=int, default=300,
                        help='Debounce time in seconds (default: 300)')
    
    args = parser.parse_args()
    
    sync = SmartSync()
    sync.config['debounce_seconds'] = args.debounce
    
    if args.command == 'watch':
        sync.start_watching()
    elif args.command == 'sync':
        result = sync.sync_now()
        print(f"Synced {result['synced']} files")
    elif args.command == 'full-sync':
        result = sync.full_sync()
        print(f"Full sync complete: {result.get('synced', 0)} files synced")
    elif args.command == 'status':
        print(f"Pending changes: {len(sync.state.pending_changes)}")
        for f, t in sync.state.pending_changes.items():
            print(f"  {f}: {t}")


if __name__ == '__main__':
    main()
