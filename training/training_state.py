"""
Training State Manager
Persistent state tracking for the LoRA retraining pipeline.
Uses atomic writes (write temp → rename) for crash safety.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

STATE_FILE = Path.home() / ".personal-ai" / "training_state.json"

VALID_STATUSES = [
    "idle",
    "collecting_data",
    "preparing_gpu",
    "training",
    "merging",
    "converting",
    "quantizing",
    "deploying",
    "complete",
    "failed",
    "interrupted"
]

DEFAULT_STATE = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "step": None,
    "step_detail": None,
    "checkpoint_path": None,
    "previous_model": None,
    "new_model": None,
    "error": None,
    "progress_pct": 0,
    "data_examples": 0,
    "orchestrator_pid": None
}


def load_state():
    """Load training state from disk, returning defaults if missing/corrupt."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            # Merge with defaults for any missing keys
            merged = {**DEFAULT_STATE, **state}
            return merged
    except (json.JSONDecodeError, IOError):
        pass
    return dict(DEFAULT_STATE)


def save_state(state):
    """Atomically save training state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file first, then rename for crash safety
    fd, tmp_path = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def update_state(**kwargs):
    """Update specific fields in the training state."""
    state = load_state()
    state.update(kwargs)
    save_state(state)
    return state


def set_status(status, detail=None, progress=None, error=None):
    """Convenience method to update status with optional detail and progress."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Must be one of {VALID_STATUSES}")
    updates = {"status": status, "step": status, "step_detail": detail}
    if progress is not None:
        updates["progress_pct"] = progress
    if error is not None:
        updates["error"] = error
    if status == "idle":
        updates["progress_pct"] = 0
        updates["step_detail"] = None
        updates["error"] = None
        updates["orchestrator_pid"] = None
    return update_state(**updates)


def start_training(previous_model):
    """Mark training as started."""
    return update_state(
        status="collecting_data",
        step="collecting_data",
        started_at=datetime.now().isoformat(),
        completed_at=None,
        previous_model=previous_model,
        new_model=None,
        error=None,
        progress_pct=0,
        data_examples=0,
        orchestrator_pid=os.getpid()
    )


def complete_training(new_model):
    """Mark training as successfully completed."""
    return update_state(
        status="complete",
        step="complete",
        step_detail="Training complete",
        completed_at=datetime.now().isoformat(),
        new_model=new_model,
        progress_pct=100,
        error=None,
        orchestrator_pid=None
    )


def fail_training(error_msg):
    """Mark training as failed."""
    return update_state(
        status="failed",
        step="failed",
        step_detail=error_msg,
        completed_at=datetime.now().isoformat(),
        error=error_msg,
        orchestrator_pid=None
    )


def reset_state():
    """Reset to idle state."""
    save_state(dict(DEFAULT_STATE))
