#!/usr/bin/env python3
"""
Settings Manager - Persistent settings for the Personal AI Framework
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess

CONFIG_DIR = Path.home() / ".personal-ai"
SETTINGS_FILE = CONFIG_DIR / "settings.json"

DEFAULT_SETTINGS = {
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
    "email_providers": {
        "gmail": {
            "configured": False,
            "authenticated": False
        },
        "outlook": {
            "configured": False,
            "authenticated": False
        }
    },
    "model": {
        "context_length": 16384
    }
}


def load_settings() -> Dict[str, Any]:
    """Load settings from file."""
    CONFIG_DIR.mkdir(exist_ok=True)
    
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            # Merge with defaults (in case new settings added)
            settings = DEFAULT_SETTINGS.copy()
            for key in saved:
                if key in settings:
                    if isinstance(settings[key], dict):
                        settings[key].update(saved[key])
                    else:
                        settings[key] = saved[key]
            return settings
        except:
            pass
    
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> bool:
    """Save settings to file."""
    CONFIG_DIR.mkdir(exist_ok=True)
    
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def get_setting(path: str) -> Any:
    """Get a specific setting by dot-path (e.g., 'digest.enabled')."""
    settings = load_settings()
    keys = path.split('.')
    value = settings
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def set_setting(path: str, value: Any) -> bool:
    """Set a specific setting by dot-path."""
    settings = load_settings()
    keys = path.split('.')
    target = settings
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value
    return save_settings(settings)


def update_email_provider_status() -> Dict:
    """Check actual status of email providers."""
    gmail_creds = CONFIG_DIR / "gmail_credentials.json"
    gmail_token = CONFIG_DIR / "gmail_token.pickle"
    outlook_creds = CONFIG_DIR / "outlook_credentials.json"
    outlook_token = CONFIG_DIR / "outlook_token.json"
    
    return {
        "gmail": {
            "configured": gmail_creds.exists(),
            "authenticated": gmail_token.exists()
        },
        "outlook": {
            "configured": outlook_creds.exists(),
            "authenticated": outlook_token.exists()
        }
    }


def setup_digest_cron(enabled: bool, time: str, email: str) -> bool:
    """Setup or remove digest cron job."""
    script_path = Path(__file__).parent / "daily_digest.py"
    
    try:
        # Get existing crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        existing = result.stdout if result.returncode == 0 else ""
        
        # Remove old digest entries
        lines = [l for l in existing.split('\n') if 'daily_digest.py' not in l and l.strip()]
        
        if enabled and email:
            hour, minute = time.split(":")
            cron_line = f'{minute} {hour} * * * cd {script_path.parent.parent} && /usr/bin/python3 {script_path} -t "{email}" -m mailto -q'
            lines.append(cron_line)
        
        # Install new crontab
        new_crontab = '\n'.join(lines) + '\n' if lines else ''
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE)
        process.communicate(new_crontab.encode())
        
        return True
    except Exception as e:
        print(f"Error setting up cron: {e}")
        return False


def apply_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply settings and return status."""
    results = {
        "saved": False,
        "digest_cron": None,
        "errors": []
    }
    
    # Save settings
    if save_settings(settings):
        results["saved"] = True
    else:
        results["errors"].append("Failed to save settings")
    
    # Setup digest cron
    digest = settings.get("digest", {})
    if setup_digest_cron(
        digest.get("enabled", False),
        digest.get("time", "08:00"),
        digest.get("email", "")
    ):
        results["digest_cron"] = "configured" if digest.get("enabled") else "disabled"
    else:
        results["errors"].append("Failed to configure digest cron")
    
    return results


# API-friendly functions
def get_all_settings() -> Dict[str, Any]:
    """Get all settings with live status updates."""
    settings = load_settings()
    settings["email_providers"] = update_email_provider_status()
    return settings
