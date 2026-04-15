#!/usr/bin/env python3
"""
Calendar Service - Google Calendar integration (per-user).

Reuses gmail_credentials.json (same Google OAuth client works for any Google
API as long as Calendar API is enabled). Each user has their own credentials
and token under ~/.personal-ai/users/<username>/.
"""
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

from user_paths import email_creds, calendar_token, user_dir


SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


def creds_path(username: str) -> Path:
    return email_creds(username)


def token_path(username: str) -> Path:
    return calendar_token(username)


def oauth_state_path(username: str) -> Path:
    return user_dir(username) / "calendar_oauth_state.json"


def is_connected(username: str) -> bool:
    return token_path(username).exists()


def is_configured(username: str) -> bool:
    return creds_path(username).exists()


def _load_creds(username: str):
    from google.auth.transport.requests import Request

    tok = token_path(username)
    with open(tok, "rb") as f:
        creds = pickle.load(f)

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(tok, "wb") as f:
            pickle.dump(creds, f)

    return creds


def _service(username: str):
    from googleapiclient.discovery import build
    return build("calendar", "v3", credentials=_load_creds(username))


def list_events(username: str, days: int = 7, max_results: int = 25) -> List[Dict]:
    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=days)).isoformat()

    result = (
        _service(username)
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
    username: str,
    summary: str,
    start: str,
    end: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None,
) -> Dict:
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
        _service(username)
        .events()
        .insert(calendarId="primary", body=body)
        .execute()
    )

    return {
        "id": result.get("id"),
        "html_link": result.get("htmlLink"),
        "summary": result.get("summary"),
    }


def disconnect(username: str) -> None:
    for f in (token_path(username), oauth_state_path(username)):
        if f.exists():
            f.unlink()
