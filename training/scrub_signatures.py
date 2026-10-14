#!/usr/bin/env python3
"""Strip email signature blocks and redact contact details from fine-tune training data.

~12% of the email-derived training answers end with a full corporate signature block (title,
company, street address, cell number, email). The fine-tune learned it, and reproduces it
verbatim at the end of generated emails — a training-data leak of real contact details, not a
hallucination. This removes the block (keeping the sign-off name, which is voice) and redacts
any other phone numbers / email addresses as [phone] / [email].

Writes scrubbed copies; the originals are never modified (failure data is a deliverable):

    python3 training/scrub_signatures.py                     # training/data -> training/data/scrubbed
    TRAINING_DATA_DIR=training/data/scrubbed python3 training/train_lora_v4.py
"""
import json, os, re, sys

PHONE = re.compile(r"(?<!\w)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?:\s*\((?:cell|mobile|office|work|fax|m|o|c)\))?"
                   r"|(?<![\w.])1?[2-9]\d{2}[2-9]\d{6}(?![\w.])"  # bare 10-digit NANP
                   r"|(?<!\w)\(?[2-9]\d{2}\)?[-. ]?[2-9]\d{6}(?!\d)", re.I)  # area code + 7 digits
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
COMPANY = re.compile(r"\b(?:Inc|Corp|Corporation|Ltd|LLC|GmbH|Semiconductors?|Technologies)\b\.?", re.I)
TITLE = re.compile(r"\b(?:Manager|Director|Engineer|President|VP|Architect|Marketing|Officer|Lead)\b")
ADDRESS = re.compile(r"\b[A-Z]{2}\s+\d{5}\b|\b\d+\s+\w+(?:\s+\w+)*\s+(?:Expressway|Street|St|Avenue|Ave|Road|Rd|Blvd|Drive|Dr|Parkway|Pkwy|Lane|Suite)\b", re.I)
SIG_WINDOW, SHORT = 12, 8  # a signature is the last <=12 non-blank lines, each <=8 words

def _sig_field(line):
    return any(p.search(line) for p in (PHONE, EMAIL, URL, COMPANY, TITLE, ADDRESS))

def strip_signature(text):
    """Cut a trailing block of short lines that contains a phone or email, starting at its first
    signature field (so a sign-off like 'Kylef' above the block is kept)."""
    if not text:
        return text
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip()]
    tail = []
    for i in reversed(idx):
        if len(lines[i].split()) > SHORT or len(tail) >= SIG_WINDOW:
            break
        tail.insert(0, i)
    if not any(PHONE.search(lines[i]) or EMAIL.search(lines[i]) for i in tail):
        return text
    cut = next(i for i in tail if _sig_field(lines[i]))
    return "\n".join(lines[:cut]).rstrip()

def redact(text):
    return URL.sub("[url]", EMAIL.sub("[email]", PHONE.sub("[phone]", text))) if text else text

def scrub(text):
    return redact(strip_signature(text))

def _scrub_record(rec):
    """Every string field: outputs, instructions, chat turns AND metadata (a subject line can hold a number)."""
    if isinstance(rec, str):
        return scrub(rec)
    if isinstance(rec, dict):
        return {k: _scrub_record(v) for k, v in rec.items()}
    if isinstance(rec, list):
        return [_scrub_record(v) for v in rec]
    return rec

def main(src=None, dst=None):
    src = src or os.environ.get("TRAINING_DATA_DIR", "training/data")
    dst = dst or os.path.join(src, "scrubbed")
    os.makedirs(dst, exist_ok=True)
    report = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json"):
            continue
        data = json.load(open(os.path.join(src, name)))
        if not isinstance(data, list):
            continue
        out = [_scrub_record(r) for r in data]
        before = sum(bool(PHONE.search(json.dumps(r)) or EMAIL.search(json.dumps(r))) for r in data)
        after = sum(bool(PHONE.search(json.dumps(r)) or EMAIL.search(json.dumps(r))) for r in out)
        changed = sum(a != b for a, b in zip(data, out))
        json.dump(out, open(os.path.join(dst, name), "w"), indent=2, ensure_ascii=False)
        report[name] = {"records": len(data), "changed": changed, "with_contact_before": before, "with_contact_after": after}
        print(f"{name:28} {len(data):6} records  {changed:5} changed  contact details {before} -> {after}")
    json.dump(report, open(os.path.join(dst, "scrub_report.json"), "w"), indent=2)
    print(f"wrote {dst}/")

if __name__ == "__main__":
    main(*sys.argv[1:3])
