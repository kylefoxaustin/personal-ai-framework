"""
GGUF Conversion Pipeline
Converts a merged HuggingFace model to quantized GGUF format.
Auto-clones and builds llama.cpp if not present.
"""
import os
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_DIR = Path.home() / ".personal-ai" / "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"


def ensure_llama_cpp():
    """Clone and set up llama.cpp if not present."""
    if (LLAMA_CPP_DIR / "convert_hf_to_gguf.py").exists():
        print("llama.cpp already set up")
        return True

    print("Setting up llama.cpp...")
    LLAMA_CPP_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Clone
    if not LLAMA_CPP_DIR.exists():
        print("Cloning llama.cpp...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(LLAMA_CPP_DIR)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Failed to clone llama.cpp: {result.stderr}")
            return False

    # Install Python requirements for conversion
    requirements = LLAMA_CPP_DIR / "requirements.txt"
    if requirements.exists():
        print("Installing llama.cpp Python requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            capture_output=True, text=True
        )

    # Build llama-quantize
    print("Building llama.cpp (for llama-quantize)...")
    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    cmake_result = subprocess.run(
        ["cmake", "..", "-DGGML_CUDA=ON"],
        cwd=str(build_dir), capture_output=True, text=True
    )
    if cmake_result.returncode != 0:
        # Try without CUDA
        print("CUDA cmake failed, trying CPU-only build...")
        subprocess.run(
            ["cmake", ".."],
            cwd=str(build_dir), capture_output=True, text=True
        )

    make_result = subprocess.run(
        ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count() or 4), "--target", "llama-quantize"],
        cwd=str(build_dir), capture_output=True, text=True
    )
    if make_result.returncode != 0:
        print(f"Warning: llama-quantize build may have failed: {make_result.stderr[:500]}")

    return (LLAMA_CPP_DIR / "convert_hf_to_gguf.py").exists()


def find_quantize_binary():
    """Find the llama-quantize binary."""
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "llama-quantize",
        LLAMA_CPP_DIR / "llama-quantize",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def convert_hf_to_gguf(merged_model_dir: str, output_dir: str) -> str:
    """Convert HuggingFace model to F16 GGUF."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    f16_gguf = output_path / "model-f16.gguf"

    print(f"Converting {merged_model_dir} to F16 GGUF...")
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    result = subprocess.run(
        [sys.executable, str(convert_script), str(merged_model_dir),
         "--outfile", str(f16_gguf), "--outtype", "f16"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    if not f16_gguf.exists():
        raise RuntimeError("GGUF conversion produced no output file")

    size_gb = f16_gguf.stat().st_size / (1024 ** 3)
    print(f"F16 GGUF created: {f16_gguf} ({size_gb:.1f} GB)")
    return str(f16_gguf)


def quantize_gguf(f16_gguf_path: str, output_path: str, quant_type: str = "Q4_K_M") -> str:
    """Quantize F16 GGUF to a smaller quantization."""
    quantize_bin = find_quantize_binary()
    if not quantize_bin:
        raise RuntimeError(
            "llama-quantize not found. Build llama.cpp first: "
            f"cd {LLAMA_CPP_DIR}/build && cmake .. && cmake --build . --target llama-quantize"
        )

    print(f"Quantizing to {quant_type}...")
    result = subprocess.run(
        [quantize_bin, str(f16_gguf_path), str(output_path), quant_type],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed: {result.stderr}")

    output = Path(output_path)
    if not output.exists():
        raise RuntimeError("Quantization produced no output file")

    size_gb = output.stat().st_size / (1024 ** 3)
    print(f"Quantized GGUF created: {output} ({size_gb:.1f} GB)")

    if size_gb < 1.0:
        raise RuntimeError(f"Quantized model too small ({size_gb:.1f} GB) — likely corrupt")

    return str(output_path)


def convert(merged_model_dir: str, final_gguf_path: str, quant_type: str = "Q4_K_M") -> str:
    """Full conversion pipeline: HF → F16 GGUF → quantized GGUF."""
    if not ensure_llama_cpp():
        raise RuntimeError("Failed to set up llama.cpp")

    # Use a temp dir for the intermediate F16 file
    output_dir = Path(final_gguf_path).parent
    f16_path = convert_hf_to_gguf(merged_model_dir, str(output_dir))

    try:
        quantize_gguf(f16_path, final_gguf_path, quant_type)
    finally:
        # Clean up the large F16 intermediate file
        f16_file = Path(f16_path)
        if f16_file.exists():
            print(f"Cleaning up intermediate F16 file ({f16_file.stat().st_size / (1024**3):.1f} GB)...")
            f16_file.unlink()

    return final_gguf_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert HF model to quantized GGUF")
    parser.add_argument("merged_model_dir", help="Path to merged HuggingFace model directory")
    parser.add_argument("output_gguf", help="Output path for quantized GGUF file")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M)")
    parser.add_argument("--setup-only", action="store_true", help="Only set up llama.cpp, don't convert")
    args = parser.parse_args()

    if args.setup_only:
        ensure_llama_cpp()
    else:
        convert(args.merged_model_dir, args.output_gguf, args.quant)
