#!/usr/bin/env python3
"""Standalone Whisper transcription worker.

Runs in a FRESH process so Whisper's imports (torch/numba/etc.) don't collide with
the libraries the long-running server has already loaded (llama_cpp, chromadb). That
in-process collision is why /upload/transcribe reported a misleading "Whisper not
installed" even though whisper + torch + CUDA are installed and work standalone.

Usage:  python3 whisper_worker.py <audio_path> [model] [device]
Prints a single JSON object on a line after the WHISPER_JSON marker (so the caller
can skip Whisper's own stderr/progress noise).
"""
import sys, json


def main():
    if len(sys.argv) < 2:
        print("WHISPER_JSON", json.dumps({"error": "usage: whisper_worker.py <audio> [model] [device]"}))
        sys.exit(2)
    audio = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    try:
        import whisper
        m = whisper.load_model(model, device=device)
        r = m.transcribe(audio, language="en", verbose=False)
        out = {
            "text": (r.get("text") or "").strip(),
            "segments": [
                {"start": s.get("start"), "end": s.get("end"), "text": (s.get("text") or "").strip()}
                for s in r.get("segments", [])
            ],
        }
        print("WHISPER_JSON", json.dumps(out))
    except Exception as e:
        print("WHISPER_JSON", json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
