#!/usr/bin/env python3
"""Validate Qwen3.8 Q4_K GDN out_proj repack against gguf.dequantize."""

import argparse
import json

import gguf
import numpy as np
import torch

from sglang.srt.layers.quantization.gguf import (
    _xpu_dequant_q4_k,
    _xpu_repack_q4_k,
)


TARGETS = (
    "blk.14.ssm_out.weight",
    "blk.22.ssm_out.weight",
    "blk.29.ssm_out.weight",
    "blk.38.ssm_out.weight",
    "blk.45.ssm_out.weight",
)
RATIO = 3
NUM_K_HEADS = 16
HEAD_V_DIM = 128
TP_SIZE = 2


class ErrorStats:
    def __init__(self):
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.elements = 0

    def update(self, actual: torch.Tensor, expected: torch.Tensor):
        diff = (actual.float() - expected.float()).abs()
        self.max_abs = max(self.max_abs, diff.max().item())
        self.sum_abs += diff.sum().item()
        self.elements += diff.numel()

    def as_dict(self):
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.sum_abs / self.elements,
            "elements": self.elements,
        }


def _dequant_repacked(qweight: torch.Tensor, col_perm=None) -> torch.Tensor:
    ql, scale, minv = _xpu_repack_q4_k(qweight, col_perm=col_perm)
    return _xpu_dequant_q4_k(ql, scale, minv, torch.float32)


def _coarse_global_permute(raw: torch.Tensor) -> torch.Tensor:
    """Match Qwen3_5ForCausalLM._gguf_gdn_transform before TP sharding."""
    rows, packed_k = raw.shape
    nk_local = NUM_K_HEADS // TP_SIZE
    value_heads = RATIO * NUM_K_HEADS
    assert packed_k % value_heads == 0
    packed_head_span = packed_k // value_heads
    return (
        raw.reshape(rows, RATIO, TP_SIZE, nk_local * packed_head_span)
        .transpose(1, 2)
        .reshape(rows, packed_k)
        .contiguous()
    )


def _validate_chunk(raw_np: np.ndarray, tensor_type, stats):
    raw = torch.from_numpy(np.array(raw_np, copy=True))
    reference = torch.from_numpy(gguf.dequantize(raw_np, tensor_type))

    base = _dequant_repacked(raw)
    stats["base"].update(base, reference)

    full_permuted = (
        reference.reshape(-1, RATIO, NUM_K_HEADS, HEAD_V_DIM)
        .transpose(1, 2)
        .reshape(reference.shape)
        .contiguous()
    )
    coarse = _coarse_global_permute(raw)
    rank_bytes = coarse.shape[1] // TP_SIZE
    rank_elements = full_permuted.shape[1] // TP_SIZE
    col_perm = (RATIO, NUM_K_HEADS // TP_SIZE, HEAD_V_DIM)

    for rank in range(TP_SIZE):
        packed = coarse[:, rank * rank_bytes : (rank + 1) * rank_bytes]
        actual = _dequant_repacked(packed, col_perm=col_perm)
        expected = full_permuted[
            :, rank * rank_elements : (rank + 1) * rank_elements
        ]
        stats[f"rank{rank}"].update(actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Validate every row; default validates first/middle/last rows.",
    )
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--max-abs", type=float, default=5e-4)
    args = parser.parse_args()

    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(args.gguf_path).tensors}
    results = {}
    for name in TARGETS:
        tensor = tensors[name]
        rows = tensor.data.shape[0]
        stats = {key: ErrorStats() for key in ("base", "rank0", "rank1")}
        if args.all_rows:
            slices = [
                slice(start, min(start + args.row_chunk, rows))
                for start in range(0, rows, args.row_chunk)
            ]
        else:
            slices = [np.array([0, rows // 2, rows - 1])]

        for row_slice in slices:
            _validate_chunk(tensor.data[row_slice], tensor.tensor_type, stats)

        results[name] = {key: value.as_dict() for key, value in stats.items()}
        print(json.dumps({name: results[name]}, sort_keys=True), flush=True)

    worst = max(
        item["max_abs"]
        for tensor_result in results.values()
        for item in tensor_result.values()
    )
    summary = {
        "all_rows": args.all_rows,
        "col_perm": [RATIO, NUM_K_HEADS // TP_SIZE, HEAD_V_DIM],
        "max_abs_limit": args.max_abs,
        "tensors": len(results),
        "worst_max_abs": worst,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if worst > args.max_abs:
        raise SystemExit(
            f"Q4_K validation failed: worst max_abs {worst} > {args.max_abs}"
        )


if __name__ == "__main__":
    main()
