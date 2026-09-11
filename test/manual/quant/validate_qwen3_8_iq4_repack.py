#!/usr/bin/env python3
"""Validate Qwen3.8 IQ4 canonical reps against gguf.dequantize.

By default every IQ4_NL/IQ4_XS tensor is sampled at its first, middle and last
rows.  All rows of the five IQ4_XS GDN ssm_out tensors are additionally checked
for both the base layout and the TP2 coarse-shard + local col_perm path.
"""

import argparse
import json

import gguf
import numpy as np
import torch

from sglang.srt.layers.quantization.gguf import (
    _xpu_dequant_iq4,
    _xpu_repack_iq4_nl,
    _xpu_repack_iq4_xs,
)


RATIO = 3
NUM_K_HEADS = 16
HEAD_V_DIM = 128
TP_SIZE = 2
SSM_OUT_TARGETS = {
    "blk.13.ssm_out.weight",
    "blk.16.ssm_out.weight",
    "blk.17.ssm_out.weight",
    "blk.18.ssm_out.weight",
    "blk.33.ssm_out.weight",
}
IQ4_TYPES = {
    gguf.GGMLQuantizationType.IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS,
}


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


def _repack(raw: torch.Tensor, tensor_type, col_perm=None):
    if tensor_type == gguf.GGMLQuantizationType.IQ4_NL:
        return _xpu_repack_iq4_nl(raw, col_perm=col_perm)
    if tensor_type == gguf.GGMLQuantizationType.IQ4_XS:
        return _xpu_repack_iq4_xs(raw, col_perm=col_perm)
    raise AssertionError(f"unexpected tensor type {tensor_type}")


def _dequant_repacked(raw: torch.Tensor, tensor_type, col_perm=None):
    packed, scale = _repack(raw, tensor_type, col_perm=col_perm)
    return _xpu_dequant_iq4(packed, scale, torch.float32)


def _coarse_global_permute(raw: torch.Tensor) -> torch.Tensor:
    """Match Qwen3_5ForCausalLM._gguf_gdn_transform before TP sharding."""
    rows, packed_k = raw.shape
    nk_local = NUM_K_HEADS // TP_SIZE
    value_heads = RATIO * NUM_K_HEADS
    if packed_k % value_heads:
        raise ValueError(
            f"packed K={packed_k} is not divisible by {value_heads} value heads"
        )
    packed_head_span = packed_k // value_heads
    return (
        raw.reshape(rows, RATIO, TP_SIZE, nk_local * packed_head_span)
        .transpose(1, 2)
        .reshape(rows, packed_k)
        .contiguous()
    )


def _update_base(raw_np, tensor_type, stats):
    raw = torch.from_numpy(np.array(raw_np, copy=True))
    expected = torch.from_numpy(gguf.dequantize(raw_np, tensor_type))
    stats.update(_dequant_repacked(raw, tensor_type), expected)


def _update_ssm(raw_np, tensor_type, stats):
    raw = torch.from_numpy(np.array(raw_np, copy=True))
    reference = torch.from_numpy(gguf.dequantize(raw_np, tensor_type))
    stats["base"].update(_dequant_repacked(raw, tensor_type), reference)

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
        packed_raw = coarse[:, rank * rank_bytes : (rank + 1) * rank_bytes]
        actual = _dequant_repacked(
            packed_raw, tensor_type, col_perm=col_perm
        )
        expected = full_permuted[
            :, rank * rank_elements : (rank + 1) * rank_elements
        ]
        stats[f"rank{rank}"].update(actual, expected)


def _row_slices(rows, all_rows, row_chunk):
    if all_rows:
        return [
            slice(start, min(start + row_chunk, rows))
            for start in range(0, rows, row_chunk)
        ]
    return [np.array(sorted({0, rows // 2, rows - 1}))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help=(
            "Validate every row of every IQ4 tensor "
            "(default samples non-ssm tensors)."
        ),
    )
    parser.add_argument(
        "--sample-ssm-rows",
        action="store_true",
        help="Only sample ssm_out rows instead of validating all of them.",
    )
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--max-abs", type=float, default=5e-4)
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.gguf_path)
    tensors = [tensor for tensor in reader.tensors if tensor.tensor_type in IQ4_TYPES]
    type_stats = {
        "IQ4_NL": ErrorStats(),
        "IQ4_XS": ErrorStats(),
    }
    ssm_results = {}
    counts = {"IQ4_NL": 0, "IQ4_XS": 0}

    for tensor in tensors:
        type_name = tensor.tensor_type.name
        counts[type_name] += 1
        is_ssm = tensor.name in SSM_OUT_TARGETS
        all_tensor_rows = args.all_rows or (is_ssm and not args.sample_ssm_rows)
        slices = _row_slices(tensor.data.shape[0], all_tensor_rows, args.row_chunk)
        if is_ssm:
            stats = {key: ErrorStats() for key in ("base", "rank0", "rank1")}
            for row_slice in slices:
                _update_ssm(tensor.data[row_slice], tensor.tensor_type, stats)
                _update_base(tensor.data[row_slice], tensor.tensor_type,
                             type_stats[type_name])
            ssm_results[tensor.name] = {
                key: value.as_dict() for key, value in stats.items()
            }
            print(json.dumps(
                {tensor.name: ssm_results[tensor.name]}, sort_keys=True
            ), flush=True)
        else:
            for row_slice in slices:
                _update_base(tensor.data[row_slice], tensor.tensor_type,
                             type_stats[type_name])

    aggregate = {
        key: value.as_dict() for key, value in type_stats.items()
    }
    worst = max(
        [item["max_abs"] for item in aggregate.values()]
        + [
            item["max_abs"]
            for tensor_result in ssm_results.values()
            for item in tensor_result.values()
        ]
    )
    summary = {
        "all_rows": args.all_rows,
        "all_ssm_rows": not args.sample_ssm_rows,
        "col_perm": [RATIO, NUM_K_HEADS // TP_SIZE, HEAD_V_DIM],
        "counts": counts,
        "max_abs_limit": args.max_abs,
        "type_stats": aggregate,
        "worst_max_abs": worst,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if counts != {"IQ4_NL": 7, "IQ4_XS": 117}:
        raise SystemExit(f"unexpected IQ4 tensor counts: {counts}")
    if set(ssm_results) != SSM_OUT_TARGETS:
        raise SystemExit(
            f"unexpected IQ4_XS ssm_out set: {sorted(ssm_results)}"
        )
    if worst > args.max_abs:
        raise SystemExit(
            f"IQ4 validation failed: worst max_abs {worst} > {args.max_abs}"
        )


if __name__ == "__main__":
    main()
