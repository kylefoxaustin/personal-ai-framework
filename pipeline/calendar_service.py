#!/usr/bin/env python3
"""
Calendar Service - Google Calendar integration.

Reuses gmail_credentials.json (same Google OAuth client works for any
Google API, as long as the Calendar API is enabled in the Cloud project).
Stores its own token at calendar_token.pickle so Gmail auth is independent.
"""
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

CONFIG_DIR = Path.home() / ".personal-ai"
CREDS_FILE = CONFIG_DIR / "gmail_credentials.json"
TOKEN_FILE = CONFIG_DIR / "calendar_token.pickle"
OAUTH_STATE_FILE = CONFIG_DIR / "calendar_oauth_state.json"

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


def is_connected() -> bool:
    return TOKEN_FILE.exists()


def is_configured() -> bool:
    return CREDS_FILE.exists()


def _load_creds():
    from google.auth.transport.requests import Request

    with open(TOKEN_FILE, "rb") as f:
        creds = pickle.load(f)

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)

    return creds


def _service():
    from googleapiclient.discovery import build

    return build("calendar", "v3", credentials=_load_creds())


def list_events(days: int = 7, max_results: int = 25) -> List[Dict]:
    """Return events from now through `days` ahead."""
    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=days)).isoformat()

    result = (
        _service()
        .events()
        .list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = []
    for e in result.get("items", []):
        start = e["start"].get("dateTime") or e["start"].get("date")
        end = e["end"].get("dateTime") or e["end"].get("date")
        events.append(
            {
                "id": e.get("id"),
                "summary": e.get("summary", "(no title)"),
                "start": start,
                "end": end,
                "location": e.get("location"),
                "description": e.get("description"),
                "html_link": e.get("htmlLink"),
                "all_day": "date" in e["start"],
            }
        )
    return events


def create_event(
    summary: str,
    start: str,
    end: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None,
) -> Dict:
    """
    Create an event. `start` and `end` must be ISO-8601 strings.
    Use YYYY-MM-DD for all-day events, or full RFC3339 with timezone for timed.
    """
    body: Dict = {"summary": summary}

    if "T" in start:
        body["start"] = {"dateTime": start}
        body["end"] = {"dateTime": end}
    else:
        body["start"] = {"date": start}
        body["end"] = {"date": end}

    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]

    result = (
        _service()
        .events()
        .insert(calendarId="primary", body=body)
        .execute()
    )

    return {
        "id": result.get("id"),
        "html_link": result.get("htmlLink"),
        "summary": result.get("summary"),
    }


def disconnect() -> None:
    for f in (TOKEN_FILE, OAUTH_STATE_FILE):
        if f.exists():
            f.unlink()
