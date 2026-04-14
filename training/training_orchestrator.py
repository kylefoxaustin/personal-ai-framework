#!/usr/bin/env python3
"""
Training Orchestrator
Coordinates the full LoRA retraining pipeline:
  1. Collect new training data from conversations
  2. Signal inference server to release GPU
  3. Run LoRA training
  4. Merge LoRA adapters into base model
  5. Convert merged model to GGUF
  6. Signal inference server to load new model
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add training dir to path for imports
TRAINING_DIR = Path(__file__).parent
PROJECT_DIR = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

from training_state import (
    load_state, set_status, start_training,
    complete_training, fail_training, update_state
)
from collect_training_data import collect as collect_data
from convert_to_gguf import convert as convert_to_gguf

API_URL = "http://localhost:8080"
SETTINGS_FILE = Path.home() / ".personal-ai" / "settings.json"
MODELS_DIR = PROJECT_DIR / "models"
LOG_DIR = TRAINING_DIR / "logs"

# Training script paths
TRAIN_SCRIPT = TRAINING_DIR / "train_lora.py"
MERGE_SCRIPT = TRAINING_DIR / "merge_lora.py"

# Interruption flag
_interrupted = False


class _Tee:
    """Duplicate writes across multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _setup_logging() -> Path:
    """Tee orchestrator stdout/stderr to a timestamped log file.

    Ensures the run is recoverable even if the controlling terminal dies.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_path = LOG_DIR / f"orchestrator_{ts}.log"
    log_file = open(log_path, "a", buffering=1)  # line-buffered text mode
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def _print_memory_baseline():
    """Print a host RAM + swap + GPU snapshot to the log.

    Captured right before training so post-mortems have starting conditions.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        print(
            f"   Memory baseline: "
            f"ram_total={vm.total / 1024**3:.1f}GB "
            f"ram_avail={vm.available / 1024**3:.1f}GB "
            f"ram_used_pct={vm.percent:.1f}% "
            f"swap={sw.used / 1024**3:.1f}GB/{sw.total / 1024**3:.1f}GB"
        )
    except ImportError:
        print("   (psutil not available — skipping RAM baseline)")
    except Exception as e:
        print(f"   (RAM baseline skipped: {e})")

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            print(
                f"   GPU baseline:    {props.name} "
                f"vram_free={free_b / 1024**3:.1f}GB/"
                f"{total_b / 1024**3:.1f}GB"
            )
    except Exception as e:
        print(f"   (GPU baseline skipped: {e})")


def signal_handler(signum, frame):
    """Handle interrupt signals from the watcher/server."""
    global _interrupted
    print("\n⚠️  Interrupt signal received — finishing current step then stopping...")
    _interrupted = True


def api_call(method, endpoint, timeout=30, **kwargs):
    """Make an HTTP call to the inference server."""
    import urllib.request
    import urllib.error

    url = f"{API_URL}{endpoint}"
    if method == "GET":
        req = urllib.request.Request(url)
    else:
        data = json.dumps(kwargs.get("json", {})).encode() if kwargs.get("json") else None
        req = urllib.request.Request(url, data=data, method=method)
        if data:
            req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  API call failed: {e}")
        return None


def load_settings():
    """Load settings from the settings file."""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {}


def save_setting(key, value):
    """Update a single top-level setting."""
    settings = load_settings()
    if "." in key:
        parts = key.split(".")
        d = settings
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    else:
        settings[key] = value
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def get_next_model_version():
    """Determine the next model version number."""
    existing = list(MODELS_DIR.glob("kyle-14b-v*-q4_k_m.gguf"))
    versions = []
    for f in existing:
        try:
            v = int(f.name.split("-v")[1].split("-")[0])
            versions.append(v)
        except (IndexError, ValueError):
            pass
    next_v = max(versions, default=2) + 1
    return next_v


def check_interrupted():
    """Check if we've been interrupted."""
    if _interrupted:
        raise InterruptedError("Training interrupted by user")


def run_orchestrator():
    """Run the full training pipeline."""
    global _interrupted
    _interrupted = False

    # Persist orchestrator output to disk immediately so a terminal kill
    # (e.g. systemd-oomd taking out the vte-spawn scope) leaves a forensic
    # trail instead of vanishing with the tab.
    log_path = _setup_logging()
    print(f"📝 Orchestrator log: {log_path}")

    # Register signal handler
    signal.signal(signal.SIGUSR1, signal_handler)

    settings = load_settings()
    training_settings = settings.get("training", {})
    min_examples = training_settings.get("min_new_examples", 50)
    epochs = training_settings.get("epochs", 3)
    lora_rank = training_settings.get("lora_rank", 64)

    # Get current model path from server
    health = api_call("GET", "/training/maintenance")
    current_model = None
    if health:
        current_model = health.get("model_path")

    print("=" * 60)
    print("🧠 Personal AI — LoRA Retraining Pipeline")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Initialize state
    start_training(current_model)

    try:
        # Step 1: Collect new training data
        print("\n📊 Step 1/6: Collecting new training data...")
        set_status("collecting_data", "Collecting conversations for training", progress=5)

        last_run = training_settings.get("last_run")
        data_result = collect_data(since=last_run)
        new_examples = data_result["new_examples"]
        total_examples = data_result["total_examples"]
        update_state(data_examples=total_examples)

        print(f"   Conversations processed: {data_result['conversations']}")
        print(f"   New examples: {new_examples}")
        print(f"   Total training examples: {total_examples}")

        if new_examples < min_examples and last_run:
            msg = f"Only {new_examples} new examples (minimum: {min_examples}). Skipping training."
            print(f"   ⚠️  {msg}")
            set_status("idle", msg)
            return

        check_interrupted()

        # Step 2: Prepare GPU (unload inference model)
        print("\n🔧 Step 2/6: Preparing GPU for training...")
        set_status("preparing_gpu", "Unloading inference model", progress=10)

        result = api_call("POST", "/training/prepare")
        if not result or result.get("status") != "ready_for_training":
            raise RuntimeError("Failed to prepare GPU — server did not confirm")
        previous_model = result.get("previous_model")
        update_state(previous_model=previous_model)
        print("   ✅ GPU ready for training")

        # Give GPU a moment to fully release memory
        time.sleep(3)
        check_interrupted()

        # Step 3: Run LoRA training
        print("\n🏋️ Step 3/6: Training LoRA adapters...")
        set_status("training", "Training LoRA adapters (this takes ~2.5 hours)", progress=15)
        _print_memory_baseline()

        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
        ]

        # Check for resume checkpoint
        state = load_state()
        if state.get("checkpoint_path") and Path(state["checkpoint_path"]).exists():
            print(f"   Resuming from checkpoint: {state['checkpoint_path']}")
            # Note: resume support requires train_lora.py modifications

        train_env = os.environ.copy()
        train_env["TRAINING_EPOCHS"] = str(epochs)
        train_env["TRAINING_LORA_RANK"] = str(lora_rank)

        proc = subprocess.Popen(
            train_cmd,
            cwd=str(PROJECT_DIR),
            env=train_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output and check for interrupts
        for line in iter(proc.stdout.readline, ""):
            print(f"   {line.rstrip()}")
            # Update progress based on training output
            if "Epoch" in line and "/" in line:
                try:
                    parts = line.split("Epoch")[1].strip().split("/")
                    current_epoch = int(parts[0].strip().rstrip(":"))
                    total_epochs = int(parts[1].strip().split()[0])
                    pct = 15 + int((current_epoch / total_epochs) * 45)
                    set_status("training", f"Training epoch {current_epoch}/{total_epochs}", progress=pct)
                except (IndexError, ValueError):
                    pass
            if _interrupted:
                print("   ⚠️  Interrupting training...")
                proc.terminate()
                proc.wait(timeout=30)
                raise InterruptedError("Training interrupted by user")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {proc.returncode}")

        print("   ✅ Training complete")
        check_interrupted()

        # Step 4: Merge LoRA into base model
        print("\n🔗 Step 4/6: Merging LoRA adapters...")
        set_status("merging", "Merging LoRA adapters into base model", progress=65)

        merge_result = subprocess.run(
            [sys.executable, str(MERGE_SCRIPT)],
            cwd=str(PROJECT_DIR),
            capture_output=True, text=True
        )
        if merge_result.returncode != 0:
            raise RuntimeError(f"Merge failed: {merge_result.stderr}")
        print("   ✅ Merge complete")
        check_interrupted()

        # Step 5: Convert to GGUF
        print("\n📦 Step 5/6: Converting to GGUF format...")
        set_status("converting", "Converting to GGUF (this may take a while)", progress=75)

        version = get_next_model_version()
        gguf_filename = f"kyle-14b-v{version}-q4_k_m.gguf"
        gguf_path = MODELS_DIR / gguf_filename
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        merged_dir = TRAINING_DIR / "output" / "merged"
        convert_to_gguf(str(merged_dir), str(gguf_path))

        if not gguf_path.exists():
            raise RuntimeError("GGUF conversion produced no output")

        size_gb = gguf_path.stat().st_size / (1024 ** 3)
        print(f"   ✅ GGUF created: {gguf_filename} ({size_gb:.1f} GB)")
        update_state(new_model=str(gguf_path))
        check_interrupted()

        # Step 6: Deploy new model
        print("\n🚀 Step 6/6: Deploying new model...")
        set_status("deploying", "Loading new model into inference server", progress=90)

        # The model path inside Docker is /app/models/...
        docker_model_path = f"/app/models/{gguf_filename}"
        result = api_call("POST", "/training/complete", timeout=120, json={"model_path": docker_model_path})

        if not result or result.get("status") != "ok":
            # Fall back to previous model
            print("   ⚠️  Failed to load new model, restoring previous...")
            api_call("POST", "/training/complete", timeout=120)
            raise RuntimeError("Failed to deploy new model")

        print("   ✅ New model deployed!")

        # Update settings
        save_setting("training.last_run", datetime.now().isoformat())
        save_setting("training.last_status", "success")

        # Mark complete
        complete_training(str(gguf_path))

        print("\n" + "=" * 60)
        print(f"✅ Training pipeline complete!")
        print(f"   New model: {gguf_filename}")
        print(f"   Training examples: {total_examples}")
        print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    except InterruptedError as e:
        print(f"\n⚠️  {e}")
        set_status("interrupted", str(e))
        save_setting("training.last_status", "interrupted")
        # Restore previous model
        print("   Restoring previous model...")
        api_call("POST", "/training/complete", timeout=120)

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        fail_training(str(e))
        save_setting("training.last_status", "failed")
        # Restore previous model
        print("   Restoring previous model...")
        api_call("POST", "/training/complete", timeout=120)
        raise


if __name__ == "__main__":
    run_orchestrator()
