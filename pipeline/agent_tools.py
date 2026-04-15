"""
Agent Tools — read-only tool registry for two-pass LLM tool use.
All tools are sandboxed and return string results.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Sandboxing constants
ALLOWED_PATHS = [
    Path("/root/knowledge"),
    Path("/root/.personal-ai"),
    Path("/app"),
]
# Write and script execution are restricted to this sandbox dir
WORKSPACE_DIR = Path("/root/.personal-ai/skippy-workspace")
MAX_FILE_SIZE = 10 * 1024  # 10 KB
MAX_WRITE_SIZE = 100 * 1024  # 100 KB
GIT_ALLOWED_SUBCOMMANDS = {"status", "log", "diff", "branch", "show"}
GIT_TIMEOUT = 10
GIT_CWD = "/app"
SCRIPT_TIMEOUT = 30
SCRIPT_ALLOWED_EXTS = {".py", ".sh"}

# Tools that require explicit user confirmation before execution
CONFIRM_TOOLS = {
    "write_file",
    "run_script",
    "schedule_reminder",
    "send_email",
    "create_calendar_event",
}


def is_confirm_tool(name: str) -> bool:
    return name in CONFIRM_TOOLS

# Tool definitions (used in the detection prompt)
TOOL_REGISTRY = {
    "read_file": {
        "description": "Read the contents of a file. Use for source code, config files, documents.",
        "parameters": {"path": "Absolute file path to read"},
    },
    "list_files": {
        "description": "List files and directories at a given path.",
        "parameters": {"path": "Absolute directory path to list"},
    },
    "web_search": {
        "description": "Search the web for current information. Use for recent events or facts beyond your training data.",
        "parameters": {"query": "Search query string"},
    },
    "git_status": {
        "description": "Run a read-only git command (status, log, diff, branch, show).",
        "parameters": {
            "subcommand": "Git subcommand: status, log, diff, branch, or show",
            "args": "(optional) Additional arguments, e.g. '--oneline -10'",
        },
    },
    "write_file": {
        "description": (
            "Create or overwrite a file in the Skippy workspace. "
            "REQUIRES USER CONFIRMATION — proposes the action, user approves before it runs. "
            "Content is the raw file body — markdown, code, plain text, etc. "
            "Do NOT include email-style 'Subject:' lines or other chat scaffolding in the content."
        ),
        "parameters": {
            "path": "Path relative to workspace, e.g. 'notes.md' or 'subdir/file.txt'",
            "content": "Full file contents to write (file format matching the extension — no email headers)",
        },
    },
    "run_script": {
        "description": (
            "Run a .py or .sh script from the Skippy workspace and return its output. "
            "REQUIRES USER CONFIRMATION. Script must already exist in the workspace."
        ),
        "parameters": {
            "path": "Path to the script relative to workspace, e.g. 'build.sh'",
            "args": "(optional) Space-separated arguments to pass to the script",
        },
    },
    "schedule_reminder": {
        "description": (
            "Create a reminder that fires at a specific future time. "
            "REQUIRES USER CONFIRMATION."
        ),
        "parameters": {
            "text": "What to remind the user about, e.g. 'Call the dentist'",
            "due_at": "When to fire, as ISO 8601 with timezone, e.g. '2026-04-15T09:00:00-05:00'",
        },
    },
    "send_email": {
        "description": (
            "Send an email via Gmail. REQUIRES USER CONFIRMATION. "
            "Write a complete, friendly email message yourself — do NOT paste raw tool output. "
            "Subject must be the subject text only — do NOT prepend 'Subject:'."
        ),
        "parameters": {
            "to": "Recipient email address",
            "subject": "Subject text only (no 'Subject:' prefix)",
            "body": "Complete email body — full sentences and a greeting. Plain text.",
        },
    },
    "create_calendar_event": {
        "description": (
            "Create a Google Calendar event on the user's primary calendar. "
            "REQUIRES USER CONFIRMATION."
        ),
        "parameters": {
            "summary": "Event title",
            "start": "ISO 8601 start (with timezone for timed events, or YYYY-MM-DD for all-day)",
            "end": "ISO 8601 end in same format as start",
            "description": "(optional) Event description / notes",
            "location": "(optional) Event location",
        },
    },
    "list_calendar_events": {
        "description": "List upcoming Google Calendar events. Safe / auto-executes.",
        "parameters": {
            "days": "(optional) How many days ahead to look, default 7",
        },
    },
}


def _is_path_allowed(path_str: str) -> bool:
    try:
        target = Path(path_str).resolve()
        return any(target == a or a in target.parents for a in ALLOWED_PATHS)
    except (ValueError, OSError):
        return False


def _is_binary(path: Path) -> bool:
    try:
        return b"\x00" in path.read_bytes()[:512]
    except OSError:
        return True


def tool_read_file(params: Dict[str, str]) -> str:
    path_str = params.get("path", "")
    if not path_str:
        return "Error: 'path' parameter is required."
    if not _is_path_allowed(path_str):
        return f"Error: Access denied. Path must be under allowed directories."

    p = Path(path_str)
    if not p.exists():
        return f"Error: File not found: {path_str}"
    if not p.is_file():
        return f"Error: Not a file: {path_str}"
    if _is_binary(p):
        return f"Error: Binary file, cannot display: {path_str}"

    size = p.stat().st_size
    content = p.read_text(errors="replace")
    if size > MAX_FILE_SIZE:
        content = content[:MAX_FILE_SIZE]
        return f"[Truncated to {MAX_FILE_SIZE} bytes — file is {size} bytes]\n\n{content}"
    return content


def tool_list_files(params: Dict[str, str]) -> str:
    path_str = params.get("path", "")
    if not path_str:
        return "Error: 'path' parameter is required."
    if not _is_path_allowed(path_str):
        return f"Error: Access denied. Path must be under allowed directories."

    p = Path(path_str)
    if not p.exists():
        return f"Error: Path not found: {path_str}"
    if not p.is_dir():
        return f"Error: Not a directory: {path_str}"

    entries = sorted(p.iterdir())
    lines = []
    for entry in entries[:100]:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"  {entry.name}{suffix}")

    header = f"Contents of {path_str} ({len(entries)} items):"
    if len(entries) > 100:
        header += " [showing first 100]"
    return header + "\n" + "\n".join(lines)


# Web search callback — set by llm_server to avoid circular import
_web_search_fn = None


def set_web_search_fn(fn):
    global _web_search_fn
    _web_search_fn = fn


def tool_web_search(params: Dict[str, str]) -> str:
    query = params.get("query", "")
    if not query:
        return "Error: 'query' parameter is required."
    if _web_search_fn is None:
        return "Error: Web search is not available."
    try:
        results = _web_search_fn(query, max_results=5)
        if not results:
            return "No search results found."
        lines = []
        for r in results:
            lines.append(f"Title: {r.get('title', '')}")
            lines.append(f"  {r.get('body', r.get('snippet', ''))}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def tool_git_status(params: Dict[str, str]) -> str:
    subcommand = params.get("subcommand", "status")
    extra_args = params.get("args", "")

    if subcommand not in GIT_ALLOWED_SUBCOMMANDS:
        return f"Error: '{subcommand}' not allowed. Use: {', '.join(sorted(GIT_ALLOWED_SUBCOMMANDS))}"

    cmd = ["git", subcommand]
    if extra_args:
        for arg in extra_args.split():
            if any(c in arg for c in [";", "|", "&", "`", "$", "(", ")"]):
                return f"Error: Forbidden characters in args: {arg}"
            cmd.append(arg)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=GIT_TIMEOUT, cwd=GIT_CWD
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output = result.stderr.strip() or f"git exited with code {result.returncode}"
        if len(output) > MAX_FILE_SIZE:
            output = output[:MAX_FILE_SIZE] + "\n[truncated]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: git command timed out after {GIT_TIMEOUT}s"
    except Exception as e:
        return f"Error running git: {e}"


def _resolve_workspace_path(rel_path: str) -> Path:
    """
    Resolve a user-supplied relative path against WORKSPACE_DIR, ensuring the
    resolved target stays inside the workspace (blocks '..' traversal).
    Raises ValueError if the path escapes the sandbox.
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = (WORKSPACE_DIR / rel_path).resolve()
    workspace_resolved = WORKSPACE_DIR.resolve()
    if workspace_resolved != candidate and workspace_resolved not in candidate.parents:
        raise ValueError("Path escapes workspace sandbox")
    return candidate


def tool_write_file(params: Dict[str, str]) -> str:
    path_str = params.get("path", "")
    content = params.get("content", "")
    if not path_str:
        return "Error: 'path' parameter is required."

    try:
        target = _resolve_workspace_path(path_str)
    except ValueError as e:
        return f"Error: {e}"

    # Strip Qwen's email-header leakage: `Subject: ...` on the first line
    # of a file body is never intentional.
    import re as _re
    content = _re.sub(r"^\s*Subject:[^\n]*\n+", "", content, count=1)

    if len(content.encode("utf-8")) > MAX_WRITE_SIZE:
        return f"Error: Content exceeds max write size ({MAX_WRITE_SIZE} bytes)."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} chars to {target}"


def tool_run_script(params: Dict[str, str]) -> str:
    path_str = params.get("path", "")
    extra_args = params.get("args", "")
    if not path_str:
        return "Error: 'path' parameter is required."

    try:
        target = _resolve_workspace_path(path_str)
    except ValueError as e:
        return f"Error: {e}"

    if not target.exists():
        return f"Error: Script not found: {target}"
    if target.suffix not in SCRIPT_ALLOWED_EXTS:
        return f"Error: Only {SCRIPT_ALLOWED_EXTS} scripts allowed."

    interpreter = "python3" if target.suffix == ".py" else "bash"
    cmd = [interpreter, str(target)]
    if extra_args:
        for arg in extra_args.split():
            if any(c in arg for c in [";", "|", "&", "`", "$", "(", ")", "<", ">"]):
                return f"Error: Forbidden characters in args: {arg}"
            cmd.append(arg)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT,
            cwd=str(WORKSPACE_DIR),
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        parts.append(f"exit code: {result.returncode}")
        combined = "\n\n".join(parts)
        if len(combined) > MAX_FILE_SIZE:
            combined = combined[:MAX_FILE_SIZE] + "\n[truncated]"
        return combined
    except subprocess.TimeoutExpired:
        return f"Error: Script timed out after {SCRIPT_TIMEOUT}s"
    except Exception as e:
        return f"Error running script: {e}"


def tool_schedule_reminder(params: Dict[str, str]) -> str:
    text = params.get("text", "").strip()
    due_at = params.get("due_at", "").strip()
    if not text:
        return "Error: 'text' parameter is required."
    if not due_at:
        return "Error: 'due_at' parameter is required (ISO 8601 with timezone)."
    try:
        import reminder_service
        result = reminder_service.schedule(text, due_at)
        return f"Reminder #{result['id']} scheduled for {result['due_at']}: {text}"
    except ValueError as e:
        return f"Error: Invalid due_at format — {e}"
    except Exception as e:
        return f"Error scheduling reminder: {e}"


def tool_send_email(params: Dict[str, str]) -> str:
    to = params.get("to", "").strip()
    subject = params.get("subject", "").strip()
    body = params.get("body", "")
    if not to or not subject:
        return "Error: 'to' and 'subject' are required."

    import pickle, base64
    from pathlib import Path as _Path
    from email.mime.text import MIMEText
    token_file = _Path("/root/.personal-ai/gmail_token.pickle")
    if not token_file.exists():
        return "Error: Gmail not connected. Open Settings → Email Providers → Setup Gmail."

    try:
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        with open(token_file, "rb") as f:
            creds = pickle.load(f)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_file, "wb") as f:
                pickle.dump(creds, f)
        service = build("gmail", "v1", credentials=creds)

        msg = MIMEText(body)
        msg["to"] = to
        msg["subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Email sent to {to} (subject: {subject})"
    except Exception as e:
        return f"Error sending email: {e}"


def tool_create_calendar_event(params: Dict[str, str]) -> str:
    try:
        import calendar_service
        if not calendar_service.is_connected():
            return "Error: Calendar not connected. Open Settings → Google Calendar → Connect."
        summary = params.get("summary", "").strip()
        start = params.get("start", "").strip()
        end = params.get("end", "").strip()
        if not (summary and start and end):
            return "Error: 'summary', 'start', and 'end' are required."
        result = calendar_service.create_event(
            summary=summary,
            start=start,
            end=end,
            description=params.get("description"),
            location=params.get("location"),
        )
        return f"Event created: {result.get('summary')} — {result.get('html_link')}"
    except Exception as e:
        return f"Error creating event: {e}"


def tool_list_calendar_events(params: Dict[str, str]) -> str:
    try:
        import calendar_service
        if not calendar_service.is_connected():
            return "Error: Calendar not connected. Open Settings → Google Calendar → Connect."
        days_raw = params.get("days", "7")
        try:
            days = int(days_raw)
        except (TypeError, ValueError):
            days = 7
        events = calendar_service.list_events(days=days, max_results=25)
        if not events:
            return f"No events in the next {days} day(s)."
        lines = [f"Upcoming events ({days} days):"]
        for e in events:
            when = e["start"]
            lines.append(f"  • {when} — {e['summary']}" + (f" @ {e['location']}" if e.get("location") else ""))
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing events: {e}"


# Dispatch table
_TOOL_FUNCTIONS = {
    "read_file": tool_read_file,
    "list_files": tool_list_files,
    "web_search": tool_web_search,
    "git_status": tool_git_status,
    "write_file": tool_write_file,
    "run_script": tool_run_script,
    "schedule_reminder": tool_schedule_reminder,
    "send_email": tool_send_email,
    "create_calendar_event": tool_create_calendar_event,
    "list_calendar_events": tool_list_calendar_events,
}


def execute_tool(name: str, params: Dict[str, str], require_safe: bool = True) -> str:
    """
    Execute a tool. By default (`require_safe=True`), tools in CONFIRM_TOOLS
    are refused — callers must first surface them to the user for approval
    and re-invoke with `require_safe=False` after confirmation.
    """
    fn = _TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'. Available: {', '.join(_TOOL_FUNCTIONS.keys())}"
    if require_safe and is_confirm_tool(name):
        return f"Error: Tool '{name}' requires user confirmation. Caller must approve first."
    return fn(params)


def get_tool_descriptions() -> str:
    lines = []
    for name, info in TOOL_REGISTRY.items():
        param_str = ", ".join(f"{k}: {v}" for k, v in info["parameters"].items())
        lines.append(f"- {name}({param_str}): {info['description']}")
    return "\n".join(lines)
