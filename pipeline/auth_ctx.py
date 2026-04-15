"""
Per-request user context.

FastAPI middleware sets `current_user` from the session cookie. Service
getters read it to route to the correct user's data store.
"""
from contextvars import ContextVar
from typing import Optional

current_user: ContextVar[Optional[str]] = ContextVar("current_user", default=None)


def get_current_username(required: bool = True) -> Optional[str]:
    u = current_user.get()
    if required and not u:
        raise RuntimeError("current_user context is not set — call from within an authenticated request")
    return u
