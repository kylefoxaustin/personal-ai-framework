"""
Multi-user authentication service.

Stores users + sessions in ~/.personal-ai/users.db. Passwords are bcrypt-hashed.
Sessions are opaque random tokens with a 30-day sliding expiration.

One user carries the `is_admin` flag — they can create/remove other users and
reset passwords. The first user created is always admin.
"""
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

import bcrypt

from user_paths import BASE_DIR

DB_PATH = BASE_DIR / "users.db"
SESSION_TTL_SECONDS = 30 * 24 * 3600


def _conn() -> sqlite3.Connection:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                must_change_password INTEGER NOT NULL DEFAULT 0,
                ai_name TEXT DEFAULT 'Skippy',
                created_at REAL NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
            """
        )


def user_count() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def list_users() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT username, is_admin, must_change_password, ai_name, created_at FROM users ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]


def get_user(username: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT username, is_admin, must_change_password, ai_name, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def create_user(username: str, password: str, is_admin: bool = False, must_change: bool = False) -> dict:
    if not username or not username.replace("_", "").replace("-", "").isalnum():
        raise ValueError("username must be alphanumeric (underscores and dashes allowed)")
    if len(password) < 6:
        raise ValueError("password must be at least 6 characters")
    # First user is automatically admin regardless of arg
    if user_count() == 0:
        is_admin = True
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with _conn() as c:
        try:
            c.execute(
                "INSERT INTO users (username, password_hash, is_admin, must_change_password, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, hashed, int(is_admin), int(must_change), time.time()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"user '{username}' already exists")
    return get_user(username)


def delete_user(username: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM users WHERE username = ?", (username,))


def set_password(username: str, new_password: str, clear_must_change: bool = True) -> None:
    if len(new_password) < 6:
        raise ValueError("password must be at least 6 characters")
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    with _conn() as c:
        c.execute(
            "UPDATE users SET password_hash = ?, must_change_password = ? WHERE username = ?",
            (hashed, 0 if clear_must_change else 1, username),
        )


def reset_password(username: str) -> str:
    """Generate a temp password, require change on next login. Return the temp password."""
    temp = secrets.token_urlsafe(9)
    set_password(username, temp, clear_must_change=False)
    with _conn() as c:
        c.execute("UPDATE users SET must_change_password = 1 WHERE username = ?", (username,))
    return temp


def set_ai_name(username: str, ai_name: str) -> None:
    with _conn() as c:
        c.execute("UPDATE users SET ai_name = ? WHERE username = ?", (ai_name or "Skippy", username))


def verify_password(username: str, password: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return False
    try:
        return bcrypt.checkpw(password.encode(), row["password_hash"].encode())
    except Exception:
        return False


# ── Sessions ──

def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (token, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, username, now, now + SESSION_TTL_SECONDS),
        )
    return token


def session_user(token: str) -> Optional[dict]:
    """Return the user dict if the token is valid and not expired; else None."""
    if not token:
        return None
    with _conn() as c:
        row = c.execute(
            """
            SELECT u.username, u.is_admin, u.must_change_password, u.ai_name, u.created_at, s.expires_at
            FROM sessions s
            JOIN users u ON u.username = s.username
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
    if not row:
        return None
    if row["expires_at"] < time.time():
        revoke_session(token)
        return None
    # Sliding expiration: push out to full TTL on each verified use
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET expires_at = ? WHERE token = ?",
            (time.time() + SESSION_TTL_SECONDS, token),
        )
    return {
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "must_change_password": bool(row["must_change_password"]),
        "ai_name": row["ai_name"] or "Skippy",
    }


def revoke_session(token: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


def revoke_all_sessions(username: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE username = ?", (username,))


# Initialize at import time so callers don't have to remember.
init_db()
