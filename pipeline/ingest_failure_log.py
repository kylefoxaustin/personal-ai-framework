"""Shared failure log for ingest scripts.

Each ingester writes one JSON line per source it gave up on. The log is
an append-only jsonl so a replay can read it back and retry. Failures
captured here represent real data loss — without this, a transient 500
or a single poison email takes down content with no record.

Entry schema:
    {"source": "<identifier>", "error": "<error string>", "ts": "<iso>"}

where <identifier> is whatever the ingester needs to re-attempt (usually
a source file path or an email relative path). Replay is per-ingester
since each script knows how to re-read its own sources.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


class FailureLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, source: str, error: str) -> None:
        entry = {
            "source": source,
            "error": error,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_sources(self) -> Iterable[str]:
        if not self.path.exists():
            return []
        sources = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sources.append(json.loads(line)["source"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return sources

    def archive(self) -> None:
        """Rename the current log so replay doesn't immediately re-fail it."""
        if self.path.exists():
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path.rename(self.path.with_suffix(f".{ts}.jsonl"))
