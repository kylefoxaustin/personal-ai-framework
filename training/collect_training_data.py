"""
Training Data Collector
Extracts new conversation data from the conversations database
and converts it to Alpaca-format training examples.
"""
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path.home() / ".personal-ai" / "conversations.db"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "train_alpaca.json"


def get_conversations_since(since: Optional[str] = None) -> List[Dict]:
    """Get conversations with messages, optionally filtered by date."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if since:
        convs = conn.execute(
            "SELECT * FROM conversations WHERE updated_at > ? ORDER BY updated_at",
            (since,)
        ).fetchall()
    else:
        convs = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at"
        ).fetchall()

    results = []
    for conv in convs:
        messages = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conv["id"],)
        ).fetchall()
        if len(messages) >= 2:
            results.append({
                "id": conv["id"],
                "title": conv["title"],
                "updated_at": conv["updated_at"],
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages]
            })

    conn.close()
    return results


def conversations_to_alpaca(conversations: List[Dict]) -> List[Dict]:
    """Convert conversations to Alpaca-format training examples.

    Each user→assistant turn pair becomes one training example.
    """
    examples = []
    for conv in conversations:
        msgs = conv["messages"]
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                user_msg = msgs[i]["content"].strip()
                assistant_msg = msgs[i + 1]["content"].strip()
                # Skip very short or empty responses
                if len(user_msg) < 10 or len(assistant_msg) < 20:
                    continue
                # Skip system messages that leaked through
                if msgs[i]["role"] == "system":
                    continue
                examples.append({
                    "instruction": user_msg,
                    "input": "",
                    "output": assistant_msg
                })
    return examples


def content_hash(example: Dict) -> str:
    """Hash an example for deduplication."""
    content = example["instruction"] + example["output"]
    return hashlib.md5(content.encode()).hexdigest()


def merge_with_existing(new_examples: List[Dict], existing_path: Path) -> List[Dict]:
    """Merge new examples with existing training data, deduplicating."""
    existing = []
    if existing_path.exists():
        with open(existing_path, "r") as f:
            existing = json.load(f)

    # Deduplicate existing data (cleans up legacy duplicates)
    seen_hashes = set()
    deduped = []
    for ex in existing:
        h = content_hash(ex)
        if h not in seen_hashes:
            deduped.append(ex)
            seen_hashes.add(h)
    existing = deduped

    added = 0
    for ex in new_examples:
        h = content_hash(ex)
        if h not in seen_hashes:
            existing.append(ex)
            seen_hashes.add(h)
            added += 1

    return existing, added


def collect(since: Optional[str] = None, output_path: Optional[Path] = None) -> Dict:
    """Main collection function. Returns stats about what was collected."""
    output_path = output_path or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get conversations
    conversations = get_conversations_since(since)
    if not conversations:
        return {"conversations": 0, "new_examples": 0, "total_examples": 0}

    # Convert to training format
    new_examples = conversations_to_alpaca(conversations)
    if not new_examples:
        return {"conversations": len(conversations), "new_examples": 0, "total_examples": 0}

    # Merge with existing data
    merged, added = merge_with_existing(new_examples, output_path)

    # Save merged dataset
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return {
        "conversations": len(conversations),
        "new_examples": added,
        "total_examples": len(merged)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect training data from conversations")
    parser.add_argument("--since", help="Only collect conversations after this ISO timestamp")
    parser.add_argument("--output", help="Output path for training data JSON")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    result = collect(since=args.since, output_path=output)
    print(f"Conversations processed: {result['conversations']}")
    print(f"New examples added: {result['new_examples']}")
    print(f"Total training examples: {result['total_examples']}")
