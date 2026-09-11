#!/usr/bin/env python3
"""Validate canonical IQ3_S repacking against gguf.dequantize."""

import argparse
import json

import gguf
import numpy as np
import torch

from sglang.srt.layers.quantization.gguf import (
    _xpu_dequant_iq3_s,
    _xpu_repack_iq3_s,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--row-chunk", type=int, default=128)
    # Final subscales are rounded to FP16.
    parser.add_argument("--max-abs", type=float, default=1.2e-4)
    args = parser.parse_args()

    tensors = [
        tensor
        for tensor in gguf.GGUFReader(args.gguf_path).tensors
        if tensor.tensor_type == gguf.GGMLQuantizationType.IQ3_S
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
        max_rel = 0.0
        dot = norm_a = norm_b = 0.0
        for start in range(0, len(indices), args.row_chunk):
            selected = indices[start : start + args.row_chunk]
            raw_np = np.array(tensor.data[selected], copy=True)
            raw = torch.from_numpy(raw_np)
            qs, qh, signs, scale = _xpu_repack_iq3_s(raw)
            device_rep = _xpu_repack_iq3_s(raw.to("xpu"))
            for cpu, device in zip((qs, qh, signs, scale), device_rep):
                torch.testing.assert_close(cpu, device.cpu(), rtol=0, atol=0)
            actual = _xpu_dequant_iq3_s(qs, qh, signs, scale, torch.float32)
            expected = torch.from_numpy(
                gguf.dequantize(raw_np, tensor.tensor_type)
            )
            # Prove every difference is exactly the final-scale FP16 rounding.
            blocks = raw_np.reshape(len(selected), -1, 110)
            d = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)
            nib = np.stack((blocks[:, :, 106:110] & 15,
                            blocks[:, :, 106:110] >> 4), axis=-1)
            exact_scale = torch.from_numpy(
                (d * (1 + 2 * nib.reshape(len(selected), -1, 8)))
                .reshape(len(selected), -1))
            ref_groups = expected.reshape(len(selected), -1, 32)
            # Dividing first recovers the integer magnitude exactly.
            magnitude = ref_groups / exact_scale.unsqueeze(-1)
            rounded = torch.where(exact_scale.unsqueeze(-1) != 0,
                                  magnitude * scale.float().unsqueeze(-1), 0)
            torch.testing.assert_close(actual.reshape_as(rounded), rounded,
                                       rtol=0, atol=0)
            diff = (actual - expected).abs()
            max_rel = max(max_rel, float((diff / expected.abs().clamp_min(1e-12)).max()))
            dot += float((actual.double() * expected.double()).sum())
            norm_a += float(actual.double().square().sum())
            norm_b += float(expected.double().square().sum())
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
            "max_rel": max_rel,
            "cosine": dot / (norm_a * norm_b)**0.5,
            "cpu_xpu_bit_exact": True,
            "scale_rounding_explains_all_error": True,
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
        raise SystemExit(f"IQ3_S canonical validation failed: {summary}")


if __name__ == "__main__":
    main()
