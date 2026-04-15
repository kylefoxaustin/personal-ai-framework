"""
Per-user filesystem paths.

All user-scoped data lives under ~/.personal-ai/users/<username>/. Shared data
(knowledge base ChromaDB, model weights, LoRA checkpoints, lan_ips.txt) stays
at the top level of ~/.personal-ai/.
"""
from pathlib import Path

BASE_DIR = Path.home() / ".personal-ai"
USERS_DIR = BASE_DIR / "users"


def user_dir(username: str) -> Path:
    """Return the per-user directory, creating it if necessary."""
    d = USERS_DIR / username
    d.mkdir(parents=True, exist_ok=True)
    return d


def conversations_db(username: str) -> Path:
    return user_dir(username) / "conversations.db"


def reminders_db(username: str) -> Path:
    return user_dir(username) / "reminders.db"


def settings_path(username: str) -> Path:
    return user_dir(username) / "settings.json"


def workspace_dir(username: str) -> Path:
    d = user_dir(username) / "skippy-workspace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def email_creds(username: str) -> Path:
    return user_dir(username) / "gmail_credentials.json"


def email_token(username: str) -> Path:
    return user_dir(username) / "gmail_token.pickle"


def calendar_token(username: str) -> Path:
    return user_dir(username) / "calendar_token.pickle"


def memory_collection_name(username: str) -> str:
    # ChromaDB collection names: alphanumeric + underscore, 3-63 chars
    safe = "".join(c if c.isalnum() else "_" for c in username)[:40]
    return f"conversation_memory_{safe}"


def facts_collection_name(username: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in username)[:40]
    return f"learned_facts_{safe}"
