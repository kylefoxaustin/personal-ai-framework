#!/usr/bin/env python3
"""
Settings Manager — per-user persistent settings.

Each user's settings.json lives under ~/.personal-ai/users/<username>/settings.json.
Call `load_settings(username)` / `save_settings(username, settings)` — no
global settings file anymore.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

from user_paths import settings_path, email_creds, email_token, user_dir


def _resolve(username):
    if username is None:
        from auth_ctx import get_current_username
        return get_current_username()
    return username

DEFAULT_SETTINGS = {
    "personality": {
        "name": "Skippy",
        "prompt": "You are a helpful AI assistant.",
        "traits": []
    },
    "digest": {
        "enabled": False,
        "time": "08:00",
        "email": "",
        "last_run": None
    },
    "sync": {
        "auto_enabled": False,
        "interval_hours": 1,
        "last_sync": None
    },
    "web_search": {
        "enabled": False
    },
    "email_providers": {
        "gmail": {"configured": False, "authenticated": False},
        "outlook": {"configured": False, "authenticated": False}
    },
    "model": {
        "context_length": 16384,
        "active_model_path": None
    },
    "training": {
        "enabled": False,
        "last_run": None,
        "last_status": None,
        "min_new_examples": 50,
        "epochs": 3,
        "lora_rank": 64
    }
}


def load_settings(username: str = None) -> Dict[str, Any]:
    username = _resolve(username)
    path = settings_path(username)
    if path.exists():
        try:
            with open(path) as f:
                saved = json.load(f)
            settings = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_SETTINGS.items()}
            for key in saved:
                if key in settings and isinstance(settings[key], dict):
                    settings[key].update(saved[key])
                else:
                    settings[key] = saved[key]
            return settings
        except Exception:
            pass
    return {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_SETTINGS.items()}


def save_settings(username_or_settings=None, settings: Dict[str, Any] = None) -> bool:
    # Back-compat: allow save_settings(settings_dict) with no username arg.
    if isinstance(username_or_settings, dict) and settings is None:
        settings = username_or_settings
        username = _resolve(None)
    else:
        username = _resolve(username_or_settings)
    path = settings_path(username)
    try:
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def get_setting(path_or_user: str, path: str = None) -> Any:
    # Back-compat: get_setting("dot.path") with no explicit user.
    if path is None:
        path = path_or_user
        username = _resolve(None)
    else:
        username = _resolve(path_or_user)
    settings = load_settings(username)
    keys = path.split('.')
    value = settings
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def set_setting(path_or_user: str, path_or_value, value=None) -> bool:
    # Back-compat: set_setting("dot.path", value) with no explicit user.
    if value is None:
        path = path_or_user
        value = path_or_value
        username = _resolve(None)
    else:
        username = _resolve(path_or_user)
        path = path_or_value
    settings = load_settings(username)
    keys = path.split('.')
    target = settings
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value
    return save_settings(username, settings)


def update_email_provider_status(username: str = None) -> Dict:
    username = _resolve(username)
    gmail_creds = email_creds(username)
    gmail_tok = email_token(username)
    outlook_creds = user_dir(username) / "outlook_credentials.json"
    outlook_token = user_dir(username) / "outlook_token.json"
    return {
        "gmail": {"configured": gmail_creds.exists(), "authenticated": gmail_tok.exists()},
        "outlook": {"configured": outlook_creds.exists(), "authenticated": outlook_token.exists()}
    }


def setup_digest_cron(username: str, enabled: bool, time: str, email: str) -> bool:
    """Setup or remove digest cron job, tagged by username so multiple users coexist."""
    script_path = Path(__file__).parent / "daily_digest.py"
    marker = f"# skippy-digest:{username}"

    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        existing = result.stdout if result.returncode == 0 else ""
        lines = [l for l in existing.split('\n') if marker not in l and l.strip()]

        if enabled and email:
            hour, minute = time.split(":")
            cron_line = (
                f'{minute} {hour} * * * cd {script_path.parent.parent} && '
                f'SKIPPY_USER="{username}" /usr/bin/python3 {script_path} -t "{email}" -m mailto -q  {marker}'
            )
            lines.append(cron_line)

        new_crontab = '\n'.join(lines) + '\n' if lines else ''
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE)
        process.communicate(new_crontab.encode())
        return True
    except Exception as e:
        print(f"Error setting up cron: {e}")
        return False


def apply_settings(username_or_settings=None, settings: Dict[str, Any] = None) -> Dict[str, Any]:
    # Back-compat: apply_settings(settings_dict).
    if isinstance(username_or_settings, dict) and settings is None:
        settings = username_or_settings
        username = _resolve(None)
    else:
        username = _resolve(username_or_settings)
    results = {"saved": False, "digest_cron": None, "errors": []}
    if save_settings(username, settings):
        results["saved"] = True
    else:
        results["errors"].append("Failed to save settings")

    digest = settings.get("digest", {})
    if setup_digest_cron(username, digest.get("enabled", False), digest.get("time", "08:00"), digest.get("email", "")):
        results["digest_cron"] = "configured" if digest.get("enabled") else "disabled"
    else:
        results["errors"].append("Failed to configure digest cron")

    return results


def get_all_settings(username: str = None) -> Dict[str, Any]:
    username = _resolve(username)
    settings = load_settings(username)
    settings["email_providers"] = update_email_provider_status(username)
    return settings
