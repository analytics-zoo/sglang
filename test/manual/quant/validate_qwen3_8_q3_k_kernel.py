#!/usr/bin/env python3
"""Validate the native Q3_K kernel on real Qwen3.8 GGUF weights."""

import argparse
import json

import gguf
import numpy as np
import torch

from custom_esimd_kernels_sglang import esimd_gemv_q3_k, esimd_gemv_q3_k_m
from sglang.srt.layers.quantization.gguf import (
    _xpu_dequant_q3_k,
    _xpu_repack_q3_k,
)


# Cover both matrix orientations present in the model.
REPRESENTATIVE_TARGETS = (
    "blk.0.ffn_up.weight",    # N=17408, K=5120
    "blk.13.ffn_down.weight",  # N=5120, K=17408
)
M_VALUES = (1, 2, 4, 8, 16)


def _sample_rows(rows: int, rows_per_region: int) -> np.ndarray:
    width = min(rows_per_region, rows)
    starts = (0, max(0, rows // 2 - width // 2), max(0, rows - width))
    return np.array(
        sorted({row for start in starts for row in range(start, start + width)})
    )


def _validate_device_repack(raw: torch.Tensor):
    expected = _xpu_repack_q3_k(raw)
    actual = _xpu_repack_q3_k(raw.to("xpu"))
    torch.xpu.synchronize()
    for device_tensor, cpu_tensor in zip(actual, expected):
        torch.testing.assert_close(device_tensor.cpu(), cpu_tensor, rtol=0, atol=0)
    return expected


def _run_kernel(ql, qh, scale, m: int, seed: int):
    dense = _xpu_dequant_q3_k(ql, qh, scale, torch.float32)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((m, dense.shape[1]), generator=generator).to(torch.float16)
    expected = (x.float() @ dense.t()).to(torch.float16)

    output = torch.empty(
        (m, dense.shape[0]), dtype=torch.float16, device="xpu"
    )
    args = (
        x.to("xpu"),
        ql.to("xpu"),
        qh.to("xpu"),
        scale.to("xpu"),
        output,
    )
    if m == 1:
        esimd_gemv_q3_k(*args)
    else:
        esimd_gemv_q3_k_m(*args)
    torch.xpu.synchronize()

    actual = output.cpu()
    diff = (actual.float() - expected.float()).abs()
    return {
        "m": m,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--rows-per-region", type=int, default=32)
    parser.add_argument("--max-abs", type=float, default=0.01)
    args = parser.parse_args()
    if args.rows_per_region < 1:
        raise SystemExit("--rows-per-region must be positive")

    tensors = {
        tensor.name: tensor for tensor in gguf.GGUFReader(args.gguf_path).tensors
    }
    results = {}
    total_rows_checked = 0
    for target_index, name in enumerate(REPRESENTATIVE_TARGETS):
        tensor = tensors[name]
        if tensor.tensor_type != gguf.GGMLQuantizationType.Q3_K:
            raise AssertionError(f"{name} is {tensor.tensor_type}, expected Q3_K")

        indices = _sample_rows(tensor.data.shape[0], args.rows_per_region)
        total_rows_checked += len(indices)
        raw = torch.from_numpy(np.array(tensor.data[indices], copy=True))
        ql, qh, scale = _validate_device_repack(raw)
        stats = [
            _run_kernel(ql, qh, scale, m, 3000 + target_index * 100 + m)
            for m in M_VALUES
        ]
        results[name] = stats
        print(
            json.dumps(
                {
                    "tensor": name,
                    "shape": [tensor.data.shape[0], int(tensor.shape[0])],
                    "rows_checked": len(indices),
                    "kernel_cases": stats,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    all_stats = [stat for stats in results.values() for stat in stats]
    worst = max(stat["max_abs"] for stat in all_stats)
    summary = {
        "tensors": len(results),
        "rows_checked": total_rows_checked,
        "m_values": list(M_VALUES),
        "kernel_cases": len(all_stats),
        "worst_max_abs": worst,
        "max_abs_limit": args.max_abs,
        "all_finite": all(stat["finite"] for stat in all_stats),
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if not summary["all_finite"] or worst > args.max_abs:
        raise SystemExit(f"Q3_K kernel validation failed: {summary}")


if __name__ == "__main__":
    main()
