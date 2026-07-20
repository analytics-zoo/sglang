#!/usr/bin/env python3
"""Qualify fused Onyx QKV + output-gate FP8 GEMV for decode M=1."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from custom_esimd_kernels_sglang import (
    esimd_gemv_fp8_pert,
    esimd_gemv_fp8_pert_fused2,
)


K = 6656
QKV_N = 2304
GATE_N = 2048


def synchronize() -> None:
    torch.xpu.synchronize()


def median_us(fn: Callable[[], None], warmup: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device("xpu")
    torch.manual_seed(0)
    x = torch.randn(1, K, dtype=torch.float16, device=device)
    qkv_weight = torch.randn(QKV_N, K, dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    gate_weight = torch.randn(GATE_N, K, dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    qkv_scale = torch.tensor(0.75, dtype=torch.float32, device=device)
    gate_scale = torch.tensor(1.25, dtype=torch.float32, device=device)
    qkv_separate = torch.empty(1, QKV_N, dtype=torch.float16, device=device)
    gate_separate = torch.empty(1, GATE_N, dtype=torch.float16, device=device)
    qkv_fused = torch.empty_like(qkv_separate)
    gate_fused = torch.empty_like(gate_separate)

    def separate() -> None:
        esimd_gemv_fp8_pert(x, qkv_weight, qkv_scale, qkv_separate)
        esimd_gemv_fp8_pert(x, gate_weight, gate_scale, gate_separate)

    def fused() -> None:
        esimd_gemv_fp8_pert_fused2(
            x,
            qkv_weight,
            qkv_scale,
            qkv_fused,
            gate_weight,
            gate_scale,
            gate_fused,
        )

    separate()
    fused()
    synchronize()
    qkv_max_abs = (qkv_fused.float() - qkv_separate.float()).abs().max().item()
    gate_max_abs = (
        gate_fused.float() - gate_separate.float()
    ).abs().max().item()
    separate_us = median_us(separate, args.warmup, args.trials)
    fused_us = median_us(fused, args.warmup, args.trials)
    report = {
        "shape": {
            "input": [1, K],
            "qkv_weight": [QKV_N, K],
            "gate_weight": [GATE_N, K],
        },
        "independent_scales": [qkv_scale.item(), gate_scale.item()],
        "qkv_max_abs": qkv_max_abs,
        "gate_max_abs": gate_max_abs,
        "separate_us": separate_us,
        "fused_us": fused_us,
        "speedup": separate_us / fused_us,
        "warmup": args.warmup,
        "trials": args.trials,
    }
    print(json.dumps(report), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
