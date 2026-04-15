"""
One-time migration: flat ~/.personal-ai/* → ~/.personal-ai/users/kyle/*.

Runs idempotently at server startup. Only moves files if:
  1. `users/` directory does not exist yet, AND
  2. the legacy flat layout has real data (conversations.db, settings.json, etc).

After migration, the bootstrap flow can create a user named 'kyle' to take
ownership of the migrated data. If the admin chooses a different username
during bootstrap, the 'kyle' data will still exist on disk and can be
reassigned by renaming the directory.
"""
import shutil
from pathlib import Path

BASE = Path.home() / ".personal-ai"
USERS = BASE / "users"
LEGACY_OWNER = "kyle"

LEGACY_FILES = [
    "conversations.db",
    "reminders.db",
    "settings.json",
    "digest_history.json",
    "gmail_credentials.json",
    "gmail_token.pickle",
    "gmail_oauth_state.json",
    "outlook_credentials.json",
    "outlook_token.json",
    "calendar_token.pickle",
    "calendar_oauth_state.json",
]
LEGACY_DIRS = ["skippy-workspace"]


def needs_migration() -> bool:
    if USERS.exists():
        return False
    # Only migrate if there's actual data to move.
    for name in LEGACY_FILES + LEGACY_DIRS:
        if (BASE / name).exists():
            return True
    return False


def migrate() -> dict:
    """Move legacy flat layout into users/<LEGACY_OWNER>/. Return a summary."""
    if not needs_migration():
        return {"migrated": False, "reason": "no-op"}

    target = USERS / LEGACY_OWNER
    target.mkdir(parents=True, exist_ok=True)

    moved_files = []
    moved_dirs = []

    for name in LEGACY_FILES:
        src = BASE / name
        if src.exists() and src.is_file():
            dst = target / name
            shutil.move(str(src), str(dst))
            moved_files.append(name)

    for name in LEGACY_DIRS:
        src = BASE / name
        if src.exists() and src.is_dir():
            dst = target / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            moved_dirs.append(name)

    summary = {
        "migrated": True,
        "owner": LEGACY_OWNER,
        "files": moved_files,
        "dirs": moved_dirs,
        "target": str(target),
    }
    print(f"[migrate] Moved {len(moved_files)} files + {len(moved_dirs)} dirs into {target}")
    print(f"[migrate] Legacy data now belongs to user '{LEGACY_OWNER}'. "
          f"Bootstrap the admin account with username='{LEGACY_OWNER}' to own it.")
    return summary


if __name__ == "__main__":
    print(migrate())
