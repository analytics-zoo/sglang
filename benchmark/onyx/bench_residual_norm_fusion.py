#!/usr/bin/env python3
"""Qualify decode-only residual/RMSNorm fusion for Onyx hidden size 6656."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from custom_esimd_kernels_sglang import (
    esimd_norm_add_norm,
    esimd_rmsnorm_residual_scalar,
)
from sglang.srt.layers.layernorm import Gemma4RMSNorm


HIDDEN_SIZE = 6656
EPS = 1e-6


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


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    denom = torch.sqrt(torch.mean(expected_f.square())).clamp_min(1e-12)
    return {
        "max_abs": diff.abs().max().item(),
        "relative_rmse": (torch.sqrt(torch.mean(diff.square())) / denom).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual_f.reshape(1, -1), expected_f.reshape(1, -1)
        ).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device("xpu")
    torch.manual_seed(0)
    attn_output = torch.randn(1, HIDDEN_SIZE, dtype=torch.float16, device=device)
    mlp_output = torch.randn(1, HIDDEN_SIZE, dtype=torch.float16, device=device)
    residual = torch.randn(1, HIDDEN_SIZE, dtype=torch.float16, device=device)

    post_attn_norm = Gemma4RMSNorm(
        HIDDEN_SIZE, eps=EPS, scale_shift=1.0
    ).to(device)
    pre_ffn_norm = Gemma4RMSNorm(HIDDEN_SIZE, eps=EPS, scale_shift=1.0).to(device)
    post_ffn_norm = Gemma4RMSNorm(HIDDEN_SIZE, eps=EPS, scale_shift=1.0).to(device)
    with torch.no_grad():
        post_attn_norm.weight.normal_(mean=0.0, std=0.1)
        pre_ffn_norm.weight.normal_(mean=0.0, std=0.1)
        post_ffn_norm.weight.normal_(mean=0.0, std=0.1)

    post_attn_weight = (post_attn_norm.weight + 1.0).half().contiguous()
    pre_ffn_weight = (pre_ffn_norm.weight + 1.0).half().contiguous()
    post_ffn_weight = post_ffn_norm.weight.half().contiguous()
    residual_work = torch.empty_like(residual)
    norm_add_norm_output = torch.empty_like(residual)
    residual_norm_output = torch.empty_like(residual)

    def norm_add_norm_baseline() -> torch.Tensor:
        residual_work.copy_(residual)
        residual_new = residual_work + post_attn_norm(attn_output)
        return pre_ffn_norm(residual_new)

    def norm_add_norm_fused() -> torch.Tensor:
        residual_work.copy_(residual)
        esimd_norm_add_norm(
            attn_output,
            residual_work,
            post_attn_weight,
            pre_ffn_weight,
            norm_add_norm_output,
            EPS,
            EPS,
        )
        return norm_add_norm_output

    def residual_norm_baseline() -> torch.Tensor:
        return residual + post_ffn_norm(mlp_output)

    def residual_norm_fused() -> torch.Tensor:
        return esimd_rmsnorm_residual_scalar(
            mlp_output,
            post_ffn_weight,
            residual,
            residual_norm_output,
            EPS,
            1.0,
        )

    first_ref = norm_add_norm_baseline()
    first_actual = norm_add_norm_fused()
    second_ref = residual_norm_baseline()
    second_actual = residual_norm_fused()
    synchronize()

    rows = []
    for name, baseline, fused, actual, expected in (
        (
            "post_attn_add_pre_ffn_norm",
            norm_add_norm_baseline,
            norm_add_norm_fused,
            first_actual,
            first_ref,
        ),
        (
            "post_ffn_norm_add",
            residual_norm_baseline,
            residual_norm_fused,
            second_actual,
            second_ref,
        ),
    ):
        baseline_us = median_us(baseline, args.warmup, args.trials)
        fused_us = median_us(fused, args.warmup, args.trials)
        row = {
            "name": name,
            "error": error_metrics(actual, expected),
            "baseline_us": baseline_us,
            "fused_us": fused_us,
            "speedup": baseline_us / fused_us,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "shape": [1, HIDDEN_SIZE],
        "dtype": "float16",
        "warmup": args.warmup,
        "trials": args.trials,
        "results": rows,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
