#!/usr/bin/env python3
"""
Verify that VeOmni can bind the Triton fused MoE path (group_gemm_fused_moe_forward).

Run from repo root with PYTHONPATH including dFactory/VeOmni, or:
  cd /path/to/aomt && PYTHONPATH=dFactory/VeOmni:dFactory python scripts/verify_triton_moe_env.py

Exit 0 only if Triton path is selected after apply_veomni_fused_moe_patch("fused").
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(repo, "dFactory", "VeOmni"), os.path.join(repo, "dFactory")):
        if p not in sys.path:
            sys.path.insert(0, p)

    errors: list[str] = []

    import torch

    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is False (need NVIDIA driver + CUDA PyTorch).")

    try:
        import triton  # noqa: F401
    except Exception as e:
        errors.append(f"Cannot import triton: {e}. Install with: pip install 'triton>=3.0'")

    try:
        import torch_npu  # noqa: F401

        errors.append("torch_npu is installed; VeOmni disables CUDA fused MoE when NPU is present.")
    except ImportError:
        pass

    os.environ.setdefault("USE_GROUP_GEMM", "1")
    if os.environ.get("USE_GROUP_GEMM", "1") != "1":
        errors.append(f"USE_GROUP_GEMM={os.environ.get('USE_GROUP_GEMM')!r} — set to 1 for Triton group GEMM.")

    from veomni.utils.import_utils import is_fused_moe_available
    from veomni.utils.env import get_env

    print(f"torch: {torch.__version__}, cuda_available={torch.cuda.is_available()}")
    try:
        import triton

        print(f"triton: {triton.__version__}")
    except Exception:
        pass
    print(f"is_fused_moe_available(): {is_fused_moe_available()}")
    print(f"USE_GROUP_GEMM (effective): {get_env('USE_GROUP_GEMM')}")

    if errors:
        print("\n--- Issues ---")
        for line in errors:
            print(f"  - {line}")
        return 1

    from veomni.ops.fused_moe import apply_veomni_fused_moe_patch

    apply_veomni_fused_moe_patch(moe_implementation="fused")
    from veomni.ops import fused_moe

    fn = fused_moe._fused_moe_forward
    name = getattr(fn, "__name__", str(fn))
    print(f"\n_fused_moe_forward after patch: {name}")

    if name == "group_gemm_fused_moe_forward":
        print("OK: Triton fused MoE (group GEMM) path is active.")
        return 0
    if name == "eager_fused_moe_forward":
        print(
            "WARN: PyTorch eager MoE fallback is active (slow). "
            "Fix CUDA + triton + USE_GROUP_GEMM=1, or see README GPU / Triton section."
        )
        return 2
    print(f"INFO: unexpected binding: {fn!r}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
