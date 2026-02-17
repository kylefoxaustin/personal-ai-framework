#!/usr/bin/env python3
"""
Datasheet & Reference Manual Ingester
Enhanced PDF processing for technical documentation
"""
import os
import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests


def clean_text(text: str) -> str:
    """Clean text for safe ingestion."""
    # Remove null bytes and other control chars
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very long repeated chars (often from PDF artifacts)
    text = re.sub(r'(.){20,}', '\1\1\1', text)
    return text.strip()

DATASHEETS_DIR = Path(__file__).parent.parent / "knowledge" / "datasheets"
MANIFEST_FILE = DATASHEETS_DIR / ".datasheet_manifest.json"
LLM_URL = "http://localhost:8080"


@dataclass
class DatasheetInfo:
    """Extracted datasheet metadata."""
    filename: str
    part_number: str
    title: str
    manufacturer: str
    category: str  # MCU, sensor, power, etc.
    pages: int
    features: List[str]
    

def extract_part_number(text: str, filename: str) -> str:
    """Try to extract part number from text or filename."""
    # Common patterns for part numbers
    patterns = [
        r'\b(i\.?MX\s*\d+\w*)\b',  # i.MX series
        r'\b(STM32\w+)\b',  # STM32
        r'\b(LPC\d+\w*)\b',  # NXP LPC
        r'\b(MIMX\w+)\b',  # NXP MIMX
        r'\b(S32\w+)\b',  # NXP S32
        r'\b(MK\w+)\b',  # Kinetis
        r'\b(AT\d+\w+)\b',  # Atmel/Microchip
        r'\b(PIC\d+\w+)\b',  # PIC
        r'\b(ESP32\w*)\b',  # ESP32
        r'\b(RP\d+)\b',  # Raspberry Pi chips
        r'\b(BCM\d+)\b',  # Broadcom
        r'\b(TMS\d+\w+)\b',  # TI
        r'\b(MSP\d+\w+)\b',  # TI MSP
        r'\b(CC\d+\w+)\b',  # TI CC
        r'\b(MAX\d+\w*)\b',  # Maxim
        r'\b(AD\d+\w*)\b',  # Analog Devices
        r'\b(LM\d+\w*)\b',  # TI LM
        r'\b(TPS\d+\w*)\b',  # TI TPS
    ]
    
    # Try text first
    for pattern in patterns:
        match = re.search(pattern, text[:5000], re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Try filename
    filename_clean = Path(filename).stem
    for pattern in patterns:
        match = re.search(pattern, filename_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Default to cleaned filename
    return filename_clean.replace('_', ' ').replace('-', ' ')[:50]


def extract_manufacturer(text: str) -> str:
    """Try to identify manufacturer from text."""
    manufacturers = {
        'nxp': 'NXP Semiconductors',
        'freescale': 'NXP (Freescale)',
        'stmicroelectronics': 'STMicroelectronics',
        'texas instruments': 'Texas Instruments',
        'microchip': 'Microchip',
        'atmel': 'Microchip (Atmel)',
        'analog devices': 'Analog Devices',
        'maxim': 'Maxim Integrated',
        'infineon': 'Infineon',
        'renesas': 'Renesas',
        'nordic': 'Nordic Semiconductor',
        'espressif': 'Espressif',
        'broadcom': 'Broadcom',
        'qualcomm': 'Qualcomm',
        'intel': 'Intel',
        'arm': 'ARM',
        'marvell': 'Marvell',
    }
    
    text_lower = text[:10000].lower()
    for key, value in manufacturers.items():
        if key in text_lower:
            return value
    
    return "Unknown"


def extract_category(text: str, part_number: str) -> str:
    """Categorize the component type."""
    text_lower = text[:10000].lower()
    pn_lower = part_number.lower()
    
    if any(x in pn_lower for x in ['imx', 'stm32', 'lpc', 'mk', 'pic', 'esp32', 'rp2']):
        return "Microcontroller/Processor"
    if any(x in text_lower for x in ['power management', 'pmic', 'voltage regulator', 'buck', 'boost', 'ldo']):
        return "Power Management"
    if any(x in text_lower for x in ['sensor', 'accelerometer', 'gyroscope', 'magnetometer', 'temperature']):
        return "Sensor"
    if any(x in text_lower for x in ['ethernet', 'phy', 'transceiver', 'can', 'lin', 'uart']):
        return "Interface/Communication"
    if any(x in text_lower for x in ['memory', 'flash', 'eeprom', 'sram', 'dram']):
        return "Memory"
    if any(x in text_lower for x in ['amplifier', 'op-amp', 'comparator', 'adc', 'dac']):
        return "Analog"
    if any(x in text_lower for x in ['display', 'lcd', 'oled', 'driver']):
        return "Display"
    
    return "General"


def extract_features(text: str) -> List[str]:
    """Extract key features from datasheet."""
    features = []
    
    # Look for features section
    features_match = re.search(
        r'(?:features|key features|highlights)[:\s]*\n((?:[\s•\-\*]*.+\n){1,15})',
        text[:15000],
        re.IGNORECASE
    )
    
    if features_match:
        feature_text = features_match.group(1)
        lines = feature_text.strip().split('\n')
        for line in lines[:10]:
            clean = re.sub(r'^[\s•\-\*]+', '', line).strip()
            if len(clean) > 10 and len(clean) < 200:
                features.append(clean)
    
    return features[:8]


def extract_tables(page) -> str:
    """Extract tables from a PDF page."""
    tables_text = []
    
    # Try to find tables using blocks
    blocks = page.get_text("blocks")
    
    # Look for structured data (multiple columns, numbers)
    for block in blocks:
        text = block[4] if len(block) > 4 else ""
        # Heuristic: tables often have multiple numbers/units
        if re.search(r'\d+\s*(mA|mV|MHz|GHz|KB|MB|GB|ns|µs|ms|°C|V|A|W|Ω)', text):
            tables_text.append(text)
    
    return "\n".join(tables_text)


def process_datasheet(filepath: Path) -> Tuple[str, DatasheetInfo]:
    """
    Process a datasheet PDF with enhanced extraction.
    
    Returns:
        Tuple of (full_text, metadata)
    """
    print(f"📄 Processing: {filepath.name}")
    
    doc = fitz.open(filepath)
    
    text_parts = []
    tables_parts = []
    
    for page_num, page in enumerate(doc):
        # Regular text extraction
        text = page.get_text()
        text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        # Try to extract tables
        tables = extract_tables(page)
        if tables:
            tables_parts.append(f"[Page {page_num + 1} Tables]\n{tables}")
    
    doc.close()
    
    full_text = clean_text("\n\n".join(text_parts))
    tables_text = "\n\n".join(tables_parts)
    
    # Extract metadata
    part_number = extract_part_number(full_text, filepath.name)
    manufacturer = extract_manufacturer(full_text)
    category = extract_category(full_text, part_number)
    features = extract_features(full_text)
    
    # Try to get title from first page
    title_match = re.search(r'^(.{10,100})$', full_text[:2000], re.MULTILINE)
    title = title_match.group(1) if title_match else part_number
    
    info = DatasheetInfo(
        filename=filepath.name,
        part_number=part_number,
        title=title[:100],
        manufacturer=manufacturer,
        category=category,
        pages=len(text_parts),
        features=features
    )
    
    # Combine text with tables
    combined_text = full_text
    if tables_text:
        combined_text += "\n\n=== EXTRACTED TABLES ===\n\n" + tables_text
    
    print(f"   Part: {info.part_number}")
    print(f"   Manufacturer: {info.manufacturer}")
    print(f"   Category: {info.category}")
    print(f"   Pages: {info.pages}")
    
    return combined_text, info


def ingest_to_rag(text: str, info: DatasheetInfo) -> int:
    """Send datasheet to RAG knowledge base in batches."""
    try:
        # Create rich metadata
        metadata = {
            "source_file": f"datasheets/{info.filename}",
            "category": "datasheets",
            "subcategory": info.category,
            "part_number": info.part_number,
            "manufacturer": info.manufacturer,
            "title": info.title,
            "file_type": ".pdf",
            "ingested_at": datetime.now().isoformat()
        }
        
        # Add features header
        header = ""
        if info.features:
            header = f"Part: {info.part_number}\nManufacturer: {info.manufacturer}\nFeatures:\n" + \
                     "\n".join(f"- {f}" for f in info.features) + "\n\n"
        
        # Split into chunks by page markers or size
        # Max ~50KB per request to be safe
        MAX_CHUNK_SIZE = 20000
        
        chunks = []
        current_chunk = header
        
        # Split by pages
        pages = re.split(r'\[Page \d+\]', text)
        
        for page in pages:
            if len(current_chunk) + len(page) > MAX_CHUNK_SIZE:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = page
            else:
                current_chunk += page
        
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        # If still too big, force split
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > MAX_CHUNK_SIZE:
                for i in range(0, len(chunk), MAX_CHUNK_SIZE):
                    final_chunks.append(chunk[i:i+MAX_CHUNK_SIZE])
            else:
                final_chunks.append(chunk)
        
        total_added = 0
        for i, chunk in enumerate(final_chunks):
            chunk_meta = metadata.copy()
            chunk_meta["batch"] = f"{i+1}/{len(final_chunks)}"
            
            # Try with retry and smaller chunks on failure
            success = False
            current_chunk = chunk
            
            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{LLM_URL}/ingest",
                        json={
                            "content": current_chunk,
                            "metadata": chunk_meta
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        total_added += result.get("chunks_added", 0)
                        success = True
                        break
                    elif attempt < 2:
                        # Try with smaller chunk
                        current_chunk = current_chunk[:len(current_chunk)//2]
                except Exception as e:
                    if attempt == 2:
                        pass  # Silent fail after retries
            
            if not success and len(chunk) > 1000:
                # Last resort: split into tiny chunks
                for j in range(0, len(chunk), 5000):
                    mini_chunk = chunk[j:j+5000]
                    if len(mini_chunk.strip()) < 100:
                        continue
                    try:
                        resp = requests.post(
                            f"{LLM_URL}/ingest",
                            json={"content": mini_chunk, "metadata": chunk_meta},
                            timeout=30
                        )
                        if resp.status_code == 200:
                            total_added += resp.json().get("chunks_added", 0)
                    except:
                        pass
        
        return total_added
            
    except Exception as e:
        print(f"   ⚠️ Ingest error: {e}")
        return 0


def load_manifest() -> Dict:
    """Load the processing manifest."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {"processed": {}}


def save_manifest(manifest: Dict):
    """Save the processing manifest."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def ingest_datasheets(force: bool = False):
    """
    Ingest all datasheets from the datasheets folder.
    
    Args:
        force: Re-process all files even if already ingested
    """
    if not DATASHEETS_DIR.exists():
        print(f"❌ Datasheets directory not found: {DATASHEETS_DIR}")
        print(f"   Create it and add PDF files")
        return
    
    # Find all PDFs
    pdfs = list(DATASHEETS_DIR.glob("**/*.pdf"))
    if not pdfs:
        print(f"📂 No PDF files found in {DATASHEETS_DIR}")
        print(f"   Add datasheets, reference manuals, and app notes")
        return
    
    print(f"\n{'='*60}")
    print(f"📚 Datasheet Ingestion")
    print(f"{'='*60}")
    print(f"Found {len(pdfs)} PDF files\n")
    
    manifest = load_manifest()
    if force:
        manifest["processed"] = {}
    
    total_chunks = 0
    processed = 0
    skipped = 0
    
    for pdf in pdfs:
        rel_path = str(pdf.relative_to(DATASHEETS_DIR))
        
        # Check if already processed
        file_stat = pdf.stat()
        file_key = f"{rel_path}:{file_stat.st_mtime}"
        
        if rel_path in manifest["processed"] and not force:
            if manifest["processed"][rel_path].get("mtime") == file_stat.st_mtime:
                skipped += 1
                continue
        
        try:
            text, info = process_datasheet(pdf)
            chunks = ingest_to_rag(text, info)
            
            manifest["processed"][rel_path] = {
                "mtime": file_stat.st_mtime,
                "chunks": chunks,
                "part_number": info.part_number,
                "manufacturer": info.manufacturer,
                "category": info.category,
                "processed_at": datetime.now().isoformat()
            }
            
            total_chunks += chunks
            processed += 1
            print(f"   ✅ Added {chunks} chunks\n")
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    save_manifest(manifest)
    
    print(f"{'='*60}")
    print(f"✅ Complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped: {skipped}")
    print(f"   Total chunks added: {total_chunks}")
    print(f"{'='*60}\n")


def list_datasheets():
    """List all ingested datasheets."""
    manifest = load_manifest()
    
    if not manifest.get("processed"):
        print("No datasheets ingested yet.")
        print(f"Add PDFs to: {DATASHEETS_DIR}")
        return
    
    print(f"\n{'='*60}")
    print(f"📚 Ingested Datasheets")
    print(f"{'='*60}\n")
    
    for filename, info in manifest["processed"].items():
        print(f"📄 {filename}")
        print(f"   Part: {info.get('part_number', 'N/A')}")
        print(f"   Manufacturer: {info.get('manufacturer', 'N/A')}")
        print(f"   Category: {info.get('category', 'N/A')}")
        print(f"   Chunks: {info.get('chunks', 0)}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest datasheets and reference manuals"
    )
    parser.add_argument(
        "command",
        choices=["ingest", "list", "info"],
        nargs="?",
        default="ingest",
        help="Command to run"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-processing of all files"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Specific file to process (for info command)"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_datasheets(force=args.force)
    elif args.command == "list":
        list_datasheets()
    elif args.command == "info" and args.file:
        filepath = Path(args.file)
        if filepath.exists():
            text, info = process_datasheet(filepath)
            print(f"\nPart Number: {info.part_number}")
            print(f"Title: {info.title}")
            print(f"Manufacturer: {info.manufacturer}")
            print(f"Category: {info.category}")
            print(f"Pages: {info.pages}")
            print(f"Features:")
            for f in info.features:
                print(f"  - {f}")
        else:
            print(f"File not found: {filepath}")


if __name__ == "__main__":
    main()
