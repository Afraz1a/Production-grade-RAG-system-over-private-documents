"""
gpu_config.py
-------------
Centralized GPU setup for RTX 4050 (CUDA).
Import this at the top of any file that runs local models.

What runs on GPU in this project:
  - Cross-encoder re-ranker     (retrieval.py)  ← biggest speedup
  - Local embedding model       (ingest.py)     ← faster ingestion
  - Query rewriting embeddings  (chain.py)      ← minor boost

What stays on CPU / API:
  - OpenAI LLM calls            (chain.py)      ← always remote
  - BM25 search                 (retrieval.py)  ← pure Python, no GPU needed
  - ChromaDB vector ops         (retrieval.py)  ← handled internally
"""

import os
import torch

# ── Detect GPU ────────────────────────────────────────────────────────────────
def get_device() -> str:
    """
    Returns 'cuda' if RTX 4050 is available, else falls back to 'cpu'.
    Always call this instead of hardcoding 'cuda' so the code works
    on machines without a GPU too.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_gpu_info():
    """Print GPU status on startup so you can confirm CUDA is being used."""
    device = get_device()

    if device == "cuda":
        gpu_name   = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[gpu] ✓ GPU detected: {gpu_name}")
        print(f"[gpu] ✓ Total VRAM:   {total_vram:.1f} GB")
        print(f"[gpu] ✓ CUDA version: {torch.version.cuda}")
        print(f"[gpu] ✓ All local models will run on GPU\n")
    else:
        print("[gpu] ✗ No CUDA GPU found — running on CPU")
        print("[gpu]   Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads\n")

    return device


def get_vram_free_gb() -> float:
    """Returns free VRAM in GB. Useful for deciding batch sizes."""
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(0)
    return free / 1024**3


def get_optimal_batch_size() -> int:
    """
    RTX 4050 has 6GB VRAM.
    Returns a safe batch size for the cross-encoder based on free VRAM.
    """
    free_gb = get_vram_free_gb()

    if free_gb >= 4.0:
        return 64    # plenty of VRAM — large batches
    elif free_gb >= 2.0:
        return 32    # moderate
    elif free_gb >= 1.0:
        return 16    # conservative
    else:
        return 8     # very low VRAM — safe minimum


def optimize_cuda():
    """
    Apply CUDA performance tweaks for inference workloads.
    Call once at startup.
    """
    if not torch.cuda.is_available():
        return

    # Use TF32 for matrix ops — faster on Ampere+ (RTX 30/40 series) with minimal precision loss
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True

    # cuDNN auto-tuner — finds fastest convolution algorithm for your GPU
    torch.backends.cudnn.benchmark         = True

    # Deterministic mode OFF (we want speed, not reproducibility)
    torch.backends.cudnn.deterministic     = False

    print("[gpu] ✓ CUDA optimizations applied (TF32, cuDNN benchmark)")


# ── Run on import ─────────────────────────────────────────────────────────────
DEVICE = print_gpu_info()
optimize_cuda()
