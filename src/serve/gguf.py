"""
Packaging (Level 5): finished LoRA adapter -> merged model -> GGUF Q4_K_M, served by llama.cpp's OpenAI-compatible server.

The adapter is merged into the full-precision base (not the 4-bit one used for training), then quantised once for
llama.cpp. That is a different model file from the one the exam scored, so notebook 06 repeats the exam on it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

LLAMA_CPP_TAG = "v0.8.0"
CONVERT_TRANSFORMERS = "transformers==4.57.6"     # what that release's converter asks for


def _run(cmd: list[str], **kw) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{(r.stdout + r.stderr)[-3000:]}")


def merge_adapter(base_model: str, adapter_repo: str, out_dir: str) -> None:
    """Base in bf16 + adapter -> plain merged model in out_dir (weights + tokenizer + chat template)."""
    import torch
    import transformers
    from packaging.version import Version
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_kw = ({"dtype": torch.bfloat16} if Version(transformers.__version__) >= Version("4.56")
                else {"torch_dtype": torch.bfloat16})
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": 0}, **dtype_kw)
    merged = PeftModel.from_pretrained(base, adapter_repo).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    del merged, base
    torch.cuda.empty_cache()


def build_llama_cpp(path: str, cuda_arch: str) -> None:
    """Clone the pinned release and build only the two programs we use, for this GPU only (much faster)."""
    if not os.path.isdir(path):
        _run(["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_TAG, "https://github.com/ggml-org/llama.cpp", path])
    _run(["cmake", "-B", "build", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
          "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"], cwd=path)
    _run(["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count() or 2),
          "--target", "llama-quantize", "llama-server"], cwd=path)


def convert_to_f16(llama_path: str, merged_dir: str, f16_path: str, libs_dir: str) -> None:
    """The converter pins its own transformers, so it gets a private copy and the notebook's packages stay as they are."""
    if not os.path.isdir(libs_dir):
        _run([sys.executable, "-m", "pip", "install", "-q", "--target", libs_dir, CONVERT_TRANSFORMERS, "sentencepiece"])
    _run([sys.executable, os.path.join(llama_path, "convert_hf_to_gguf.py"), merged_dir,
          "--outfile", f16_path, "--outtype", "f16"], env={**os.environ, "PYTHONPATH": libs_dir})


def quantize(llama_path: str, f16_path: str, q4_path: str) -> None:
    _run([os.path.join(llama_path, "build/bin/llama-quantize"), f16_path, q4_path, "Q4_K_M"])


def start_server(llama_path: str, gguf_path: str, port: int = 8080, log_path: str = "/content/llama_server.log",
                 timeout_s: int = 300) -> subprocess.Popen:
    """llama-server on the GPU, 4 replies at a time. Returns once /health answers."""
    log = open(log_path, "w")
    proc = subprocess.Popen([os.path.join(llama_path, "build/bin/llama-server"), "-m", gguf_path, "--host", "127.0.0.1",
                             "--port", str(port), "-ngl", "99", "-c", "8192", "-np", "4", "--jinja"],
                            stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return proc
        except OSError:
            pass
        time.sleep(2)
    proc.terminate()
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        raise SystemExit("llama-server did not start:\n" + fh.read()[-3000:])
