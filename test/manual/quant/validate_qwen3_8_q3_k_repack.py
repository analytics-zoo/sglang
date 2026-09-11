#!/usr/bin/env python3
"""Validate canonical Q3_K repacking against gguf.dequantize."""

import argparse
import json

import gguf
import numpy as np
import torch

from sglang.srt.layers.quantization.gguf import (
    _xpu_dequant_q3_k,
    _xpu_repack_q3_k,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--row-chunk", type=int, default=128)
    # The canonical ABI stores final d*scale6 in FP16. Across all 109,568
    # real rows the resulting rounding peak is 6.1035e-5 (mean ~1.3e-6).
    parser.add_argument("--max-abs", type=float, default=1e-4)
    args = parser.parse_args()

    tensors = [
        tensor
        for tensor in gguf.GGUFReader(args.gguf_path).tensors
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q3_K
    ]
    results = []
    for tensor in tensors:
        rows = tensor.data.shape[0]
        if args.all_rows:
            indices = np.arange(rows)
        else:
            indices = np.array(sorted({0, rows // 2, rows - 1}))

        max_abs = 0.0
        abs_sum = 0.0
        elements = 0
        finite = True
        for start in range(0, len(indices), args.row_chunk):
            selected = indices[start : start + args.row_chunk]
            raw_np = np.array(tensor.data[selected], copy=True)
            raw = torch.from_numpy(raw_np)
            ql, qh, scale = _xpu_repack_q3_k(raw)
            actual = _xpu_dequant_q3_k(ql, qh, scale, torch.float32)
            expected = torch.from_numpy(
                gguf.dequantize(raw_np, tensor.tensor_type)
            )
            diff = (actual - expected).abs()
            max_abs = max(max_abs, float(diff.max()))
            abs_sum += float(diff.sum())
            elements += diff.numel()
            finite = finite and bool(torch.isfinite(actual).all())

        result = {
            "tensor": tensor.name,
            "rows_checked": len(indices),
            "rows_total": rows,
            "k": int(tensor.shape[0]),
            "max_abs": max_abs,
            "mean_abs": abs_sum / elements,
            "finite": finite,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "tensors": len(results),
        "rows_checked": sum(result["rows_checked"] for result in results),
        "worst_max_abs": max(result["max_abs"] for result in results),
        "max_abs_limit": args.max_abs,
        "all_finite": all(result["finite"] for result in results),
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if not summary["all_finite"] or summary["worst_max_abs"] > args.max_abs:
        raise SystemExit(f"Q3_K canonical validation failed: {summary}")


if __name__ == "__main__":
    main()
