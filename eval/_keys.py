"""Shared API-key loader for eval scripts — the durable fix for the recurring
"ANTHROPIC_API_KEY not set" problem in Claude Code sessions.

WHY THIS EXISTS. Claude Code's Bash tool (and its `!` prompt) run NON-interactive
shells that never source ~/.bashrc, so interactive-shell exports are invisible to
every script we run. And the obvious "just put it in settings.local.json env" route
is a trap: exporting ANTHROPIC_API_KEY into the GLOBAL Claude Code env has broken
Remote Control before. So keys live in ONE gitignored file OUTSIDE the repo and are
loaded into os.environ only for the process that calls ensure_keys() — scoped, not
global. Populate it once; it then works for every eval script, every session, forever.

FILE: ~/.personal-ai/keys.env   (KEY=VALUE lines; see docs/api-keys-setup.md)
It is outside the git tree entirely, so it cannot be committed by accident.
"""
import os
from pathlib import Path

KEYS_FILE = Path.home() / ".personal-ai" / "keys.env"

def ensure_keys(required=()):
    """Load ~/.personal-ai/keys.env into os.environ (only keys not already set).
    Returns the list of still-missing required keys (empty = all present)."""
    if KEYS_FILE.exists():
        for raw in KEYS_FILE.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k.startswith("export "):
                k = k[len("export "):].strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    return [k for k in required if not os.environ.get(k)]
