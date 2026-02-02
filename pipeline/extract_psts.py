#!/usr/bin/env python3
"""
PST Extraction Script with Progress Tracking
Extracts Outlook PST files to mbox format for ingestion
"""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def extract_pst(pst_path, output_dir):
    """Extract a single PST file to mbox format."""
    cmd = [
        "readpst",
        "-M",           # Output as mbox files
        "-o", str(output_dir),
        "-j", "4",      # Parallel jobs
        str(pst_path)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output
    for line in process.stdout:
        line = line.strip()
        if line:
            print(f"    {line}")
    
    process.wait()
    return process.returncode == 0

def main():
    emails_dir = Path("knowledge/emails")
    output_dir = emails_dir / "extracted"
    output_dir.mkdir(exist_ok=True)
    
    # Find all PST files
    pst_files = sorted(emails_dir.glob("*.pst"))
    
    if not pst_files:
        print("❌ No PST files found!")
        return
    
    total_size = sum(get_file_size_mb(p) for p in pst_files)
    
    print("=" * 60)
    print("📧 PST Extraction Tool")
    print("=" * 60)
    print(f"Found {len(pst_files)} PST files ({total_size:.1f} MB total)")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    successful = 0
    failed = 0
    processed_size = 0
    
    for i, pst_file in enumerate(pst_files, 1):
        size_mb = get_file_size_mb(pst_file)
        pct_complete = (processed_size / total_size) * 100 if total_size > 0 else 0
        
        print()
        print(f"[{i}/{len(pst_files)}] {pst_file.name}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Overall progress: {pct_complete:.1f}%")
        print(f"    Started: {datetime.now().strftime('%H:%M:%S')}")
        print("    Extracting...")
        
        try:
            if extract_pst(pst_file, output_dir):
                successful += 1
                print(f"    ✅ Complete!")
            else:
                failed += 1
                print(f"    ⚠️ Completed with warnings")
        except Exception as e:
            failed += 1
            print(f"    ❌ Error: {e}")
        
        processed_size += size_mb
    
    # Summary
    print()
    print("=" * 60)
    print("📊 Extraction Summary")
    print("=" * 60)
    print(f"Successful: {successful}/{len(pst_files)}")
    if failed:
        print(f"Failed: {failed}")
    
    # Count extracted mbox files
    mbox_files = list(output_dir.rglob("*.mbox")) + list(output_dir.rglob("mbox"))
    mbox_count = len(mbox_files)
    
    print(f"Mbox files created: {mbox_count}")
    print()
    print("Next step: Move/copy mbox files for ingestion or update config")
    print(f"  ls -la {output_dir}")

if __name__ == "__main__":
    main()
