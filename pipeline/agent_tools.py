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
MAX_FILE_SIZE = 10 * 1024  # 10 KB
GIT_ALLOWED_SUBCOMMANDS = {"status", "log", "diff", "branch", "show"}
GIT_TIMEOUT = 10
GIT_CWD = "/app"

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


# Dispatch table
_TOOL_FUNCTIONS = {
    "read_file": tool_read_file,
    "list_files": tool_list_files,
    "web_search": tool_web_search,
    "git_status": tool_git_status,
}


def execute_tool(name: str, params: Dict[str, str]) -> str:
    fn = _TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'. Available: {', '.join(_TOOL_FUNCTIONS.keys())}"
    return fn(params)


def get_tool_descriptions() -> str:
    lines = []
    for name, info in TOOL_REGISTRY.items():
        param_str = ", ".join(f"{k}: {v}" for k, v in info["parameters"].items())
        lines.append(f"- {name}({param_str}): {info['description']}")
    return "\n".join(lines)
