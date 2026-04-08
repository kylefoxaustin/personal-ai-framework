#!/usr/bin/env python3
"""
Training Watcher
Host-side service that watches for trigger files and launches the training orchestrator.
Run with: ./run.sh training-watch
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TRIGGER_FILE = Path.home() / ".personal-ai" / "training_trigger.json"
STATE_FILE = Path.home() / ".personal-ai" / "training_state.json"
ORCHESTRATOR = Path(__file__).parent / "training_orchestrator.py"
VENV_PYTHON = Path(__file__).parent / ".venv" / "bin" / "python"

_running_process = None


def read_trigger():
    """Read and consume the trigger file."""
    if not TRIGGER_FILE.exists():
        return None
    try:
        with open(TRIGGER_FILE) as f:
            trigger = json.load(f)
        TRIGGER_FILE.unlink()
        return trigger
    except (json.JSONDecodeError, IOError):
        try:
            TRIGGER_FILE.unlink()
        except OSError:
            pass
        return None


def get_orchestrator_pid():
    """Get the PID of a running orchestrator from state file."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                state = json.load(f)
            pid = state.get("orchestrator_pid")
            if pid and pid_exists(pid):
                return pid
    except (json.JSONDecodeError, IOError):
        pass
    return None


def pid_exists(pid):
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def start_orchestrator():
    """Launch the training orchestrator as a subprocess."""
    global _running_process

    if _running_process and _running_process.poll() is None:
        print("⚠️  Orchestrator already running, ignoring trigger")
        return

    # Use venv Python if available (isolated training dependencies)
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    print(f"🚀 Starting training orchestrator...")
    print(f"   Python: {python}")
    _running_process = subprocess.Popen(
        [python, str(ORCHESTRATOR)],
        cwd=str(ORCHESTRATOR.parent.parent),
    )
    print(f"   PID: {_running_process.pid}")


def interrupt_orchestrator():
    """Send interrupt signal to running orchestrator."""
    pid = get_orchestrator_pid()
    if pid:
        print(f"⚠️  Sending interrupt to orchestrator (PID {pid})...")
        try:
            os.kill(pid, signal.SIGUSR1)
        except (OSError, ProcessLookupError):
            print("   Orchestrator process not found")
    elif _running_process and _running_process.poll() is None:
        print(f"⚠️  Sending interrupt to orchestrator (PID {_running_process.pid})...")
        _running_process.send_signal(signal.SIGUSR1)
    else:
        print("   No running orchestrator to interrupt")


def cleanup_stale_state():
    """Check for stale training state on startup (e.g., after a crash)."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                state = json.load(f)
            status = state.get("status", "idle")
            if status not in ("idle", "complete", "failed", "interrupted"):
                pid = state.get("orchestrator_pid")
                if pid and not pid_exists(pid):
                    print(f"⚠️  Found stale training state (status={status}, dead PID={pid})")
                    print("   Resetting to idle...")
                    state["status"] = "failed"
                    state["error"] = "Training process crashed or was killed"
                    state["orchestrator_pid"] = None
                    with open(STATE_FILE, "w") as f:
                        json.dump(state, f, indent=2)
    except (json.JSONDecodeError, IOError):
        pass


AUTO_CHECK_INTERVAL = 6 * 3600  # Check every 6 hours
SETTINGS_FILE = Path.home() / ".personal-ai" / "settings.json"
CONVERSATIONS_DB = Path.home() / ".personal-ai" / "conversations.db"


def check_auto_training():
    """Check if auto-training should be triggered based on new conversation count."""
    if _running_process and _running_process.poll() is None:
        return  # Already training

    # Load settings
    try:
        if not SETTINGS_FILE.exists():
            return
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
        training = settings.get("training", {})
        if not training.get("enabled", False):
            return
        min_examples = training.get("min_new_examples", 50)
        last_run = training.get("last_run")
    except (json.JSONDecodeError, IOError):
        return

    # Count new conversations since last training
    try:
        import sqlite3
        if not CONVERSATIONS_DB.exists():
            return
        conn = sqlite3.connect(CONVERSATIONS_DB)
        if last_run:
            count = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE updated_at > ?",
                (last_run,)
            ).fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        conn.close()
    except Exception:
        return

    if count >= min_examples:
        print(f"\n🤖 Auto-training: {count} new conversations (threshold: {min_examples})")
        start_orchestrator()
    else:
        print(f"   Auto-training check: {count}/{min_examples} new conversations")


def run_watcher():
    """Main watcher loop."""
    global _running_process

    print("=" * 60)
    print("🔍 Training Watcher — monitoring for training triggers")
    print(f"   Trigger file: {TRIGGER_FILE}")
    print(f"   Orchestrator: {ORCHESTRATOR}")
    print(f"   Auto-check interval: {AUTO_CHECK_INTERVAL // 3600}h")
    print("   Press Ctrl+C to stop")
    print("=" * 60)

    cleanup_stale_state()
    last_auto_check = time.time()

    try:
        while True:
            trigger = read_trigger()
            if trigger:
                action = trigger.get("action", "start")
                print(f"\n📨 Trigger received: {action} at {trigger.get('timestamp', 'unknown')}")

                if action == "start":
                    start_orchestrator()
                elif action == "interrupt":
                    interrupt_orchestrator()
                else:
                    print(f"   Unknown action: {action}")

            # Check if running process finished
            if _running_process and _running_process.poll() is not None:
                rc = _running_process.returncode
                if rc == 0:
                    print("✅ Orchestrator finished successfully")
                else:
                    print(f"❌ Orchestrator exited with code {rc}")
                _running_process = None

            # Periodic auto-training check
            if time.time() - last_auto_check >= AUTO_CHECK_INTERVAL:
                check_auto_training()
                last_auto_check = time.time()

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n👋 Watcher stopped")
        if _running_process and _running_process.poll() is None:
            print("   Waiting for orchestrator to finish...")
            _running_process.wait(timeout=10)


if __name__ == "__main__":
    run_watcher()
