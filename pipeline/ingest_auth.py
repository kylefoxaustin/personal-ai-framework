"""Shared auth helper for ingest scripts.

Multi-user auth (v5.9+) gates the /ingest endpoints. Each ingest script
calls login_headers() once at startup and attaches the returned dict to
every requests.post(..., headers=...). Exits non-zero if credentials are
missing or login fails.
"""
import os
import sys
import requests


def login_headers(server_url: str) -> dict:
    user = os.environ.get("SKIPPY_USER")
    password = os.environ.get("SKIPPY_PASSWORD")
    if not user or not password:
        print("❌ SKIPPY_USER and SKIPPY_PASSWORD must be set — /ingest requires auth.")
        sys.exit(2)
    resp = requests.post(
        f"{server_url}/auth/login",
        json={"username": user, "password": password},
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"❌ Login failed ({resp.status_code}): {resp.text}")
        sys.exit(2)
    token = resp.json()["token"]
    print(f"🔑 Authenticated as {user}")
    return {"Authorization": f"Bearer {token}"}
