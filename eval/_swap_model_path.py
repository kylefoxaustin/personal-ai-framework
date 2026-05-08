#!/usr/bin/env python3
"""Edit pipeline/config.yaml model.path → target. Targeted line-based edit;
preserves all other formatting. Usage: _swap_model_path.py <target_path>.
Verifies the result is valid YAML before exiting."""
import sys
import yaml
from pathlib import Path

CONFIG_PATH = Path("pipeline/config.yaml")

def main():
    if len(sys.argv) != 2:
        print("usage: _swap_model_path.py <target_path>", file=sys.stderr)
        sys.exit(1)
    target = sys.argv[1]
    lines = CONFIG_PATH.read_text().splitlines()
    model_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "model:" and not line.startswith(" "):
            model_idx = i
            break
    if model_idx is None:
        print(f"ERROR: no top-level 'model:' key in {CONFIG_PATH}", file=sys.stderr)
        sys.exit(2)
    path_idx = None
    for i in range(model_idx + 1, min(model_idx + 8, len(lines))):
        if lines[i].lstrip().startswith("path:"):
            path_idx = i
            break
    if path_idx is None:
        print(f"ERROR: no 'path:' under model: in {CONFIG_PATH}", file=sys.stderr)
        sys.exit(3)
    indent = lines[path_idx][: len(lines[path_idx]) - len(lines[path_idx].lstrip())]
    lines[path_idx] = f'{indent}path: "{target}"'
    new_text = "\n".join(lines) + "\n"
    # Validate YAML before writing
    try:
        d = yaml.safe_load(new_text)
        assert d["model"]["path"] == target, "post-edit yaml round-trip mismatch"
    except Exception as e:
        print(f"ERROR: edited content is not valid YAML: {e}", file=sys.stderr)
        sys.exit(4)
    CONFIG_PATH.write_text(new_text)
    print(f"  config.yaml model.path → {target}")

if __name__ == "__main__":
    main()
