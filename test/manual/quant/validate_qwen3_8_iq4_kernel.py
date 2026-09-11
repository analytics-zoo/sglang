#!/usr/bin/env python3
"""Validate the native IQ4 kernel on representative Qwen3.8 GGUF weights."""

import argparse
import json

import gguf
import numpy as np
import torch

from custom_esimd_kernels_sglang import esimd_gemv_iq4, esimd_gemv_iq4_m
from sglang.srt.layers.quantization.gguf import (
    _xpu_dequant_iq4,
    _xpu_repack_iq4_nl,
    _xpu_repack_iq4_xs,
)


REPRESENTATIVE_TARGETS = (
    "blk.1.ffn_down.weight",     # IQ4_NL K=17408
    "blk.21.attn_qkv.weight",    # IQ4_NL K=5120
    "blk.27.ffn_gate.weight",    # IQ4_NL K=5120, different N
    "blk.0.ffn_down.weight",     # IQ4_XS K=17408
    "blk.0.ffn_gate.weight",     # IQ4_XS K=5120
    "blk.8.attn_gate.weight",    # IQ4_XS K=5120, different N
    "blk.11.attn_q.weight",      # IQ4_XS K=5120, different N
    "blk.13.ssm_out.weight",     # IQ4_XS K=6144
    "blk.36.attn_qkv.weight",    # IQ4_XS K=5120, different N
)
SSM_OUT_TARGETS = (
    "blk.13.ssm_out.weight",
    "blk.16.ssm_out.weight",
    "blk.17.ssm_out.weight",
    "blk.18.ssm_out.weight",
    "blk.33.ssm_out.weight",
)
RATIO = 3
NUM_K_HEADS = 16
HEAD_V_DIM = 128
TP_SIZE = 2


def _repack(raw, tensor_type, col_perm=None):
    if tensor_type == gguf.GGMLQuantizationType.IQ4_NL:
        return _xpu_repack_iq4_nl(raw, col_perm=col_perm)
    if tensor_type == gguf.GGMLQuantizationType.IQ4_XS:
        return _xpu_repack_iq4_xs(raw, col_perm=col_perm)
    raise AssertionError(tensor_type)


def _coarse_global_permute(raw):
    rows, packed_k = raw.shape
    nk_local = NUM_K_HEADS // TP_SIZE
    packed_head_span = packed_k // (RATIO * NUM_K_HEADS)
    return (
        raw.reshape(rows, RATIO, TP_SIZE, nk_local * packed_head_span)
        .transpose(1, 2)
        .reshape(rows, packed_k)
        .contiguous()
    )


def _run_kernel(packed_cpu, scale_cpu, m, seed):
    dense = _xpu_dequant_iq4(packed_cpu, scale_cpu, torch.float32)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((m, dense.shape[1]), generator=generator).to(torch.float16)
    expected = (x.float() @ dense.t()).to(torch.float16)

    x_xpu = x.to("xpu")
    packed_xpu = packed_cpu.to("xpu")
    scale_xpu = scale_cpu.to("xpu")
    output = torch.empty(
        (m, dense.shape[0]), dtype=torch.float16, device="xpu"
    )
    if m == 1:
        esimd_gemv_iq4(x_xpu, packed_xpu, scale_xpu, output)
    else:
        esimd_gemv_iq4_m(x_xpu, packed_xpu, scale_xpu, output)
    torch.xpu.synchronize()
    actual = output.cpu()
    diff = (actual.float() - expected.float()).abs()
    return {
        "m": m,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def _validate_device_repack(raw, tensor_type, col_perm=None):
    expected = _repack(raw, tensor_type, col_perm=col_perm)
    actual = _repack(raw.to("xpu"), tensor_type, col_perm=col_perm)
    torch.xpu.synchronize()
    for device_tensor, cpu_tensor in zip(actual, expected):
        torch.testing.assert_close(device_tensor.cpu(), cpu_tensor, rtol=0, atol=0)
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--max-abs", type=float, default=0.01)
    args = parser.parse_args()

    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(args.gguf_path).tensors}
    results = {}
    for index, name in enumerate(REPRESENTATIVE_TARGETS):
        tensor = tensors[name]
        rows = tensor.data.shape[0]
        raw = torch.from_numpy(np.array(
            tensor.data[np.array(sorted({0, rows // 2, rows - 1}))], copy=True
        ))
        packed, scale = _validate_device_repack(raw, tensor.tensor_type)
        results[name] = [_run_kernel(packed, scale, 1, 1000 + index)]

    col_perm = (RATIO, NUM_K_HEADS // TP_SIZE, HEAD_V_DIM)
    for index, name in enumerate(SSM_OUT_TARGETS):
        tensor = tensors[name]
        raw = torch.from_numpy(np.array(tensor.data, copy=True))
        coarse = _coarse_global_permute(raw)
        rank_bytes = coarse.shape[1] // TP_SIZE
        results[name + ":tp"] = []
        for rank in range(TP_SIZE):
            rank_raw = coarse[:, rank * rank_bytes : (rank + 1) * rank_bytes]
            packed, scale = _validate_device_repack(
                rank_raw, tensor.tensor_type, col_perm=col_perm
            )
            ms = (1, 2, 4, 8, 16) if index == 0 and rank == 0 else (1,)
            for m in ms:
                results[name + ":tp"].append(
                    {"rank": rank, **_run_kernel(packed, scale, m, 2000 + m)}
                )
        print(json.dumps({name: results[name + ":tp"]}), flush=True)

    all_stats = [stat for value in results.values() for stat in value]
    worst = max(stat["max_abs"] for stat in all_stats)
    summary = {
        "representative_tensors": len(REPRESENTATIVE_TARGETS),
        "ssm_out_tensors": len(SSM_OUT_TARGETS),
        "ssm_col_perm": list(col_perm),
        "kernel_cases": len(all_stats),
        "worst_max_abs": worst,
        "max_abs_limit": args.max_abs,
        "all_finite": all(stat["finite"] for stat in all_stats),
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if not summary["all_finite"] or worst > args.max_abs:
        raise SystemExit(f"IQ4 kernel validation failed: {summary}")


if __name__ == "__main__":
    main()
