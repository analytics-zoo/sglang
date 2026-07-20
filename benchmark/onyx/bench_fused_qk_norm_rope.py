#!/usr/bin/env python3
"""Qualify fused HD128 QK RMSNorm + RoPE for the Onyx TP=2 shape."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


HEAD_DIM = 128
Q_HEADS = 16
KV_HEADS = 1
Q_SIZE = Q_HEADS * HEAD_DIM
KV_SIZE = KV_HEADS * HEAD_DIM
QKV_SIZE = Q_SIZE + 2 * KV_SIZE
QUERY_SCALE = 43.7840518911 / math.sqrt(HEAD_DIM)


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


def run_case(
    token_count: int,
    use_rope: bool,
    warmup: int,
    trials: int,
) -> dict:
    from sgl_kernel import fused_qk_norm_rope

    device = torch.device("xpu")
    torch.manual_seed(0)
    source = torch.randn(
        token_count, QKV_SIZE, dtype=torch.float16, device=device
    ).contiguous()
    positions = torch.arange(token_count, dtype=torch.int32, device=device)

    q_norm = Gemma4RMSNorm(HEAD_DIM, eps=1e-6, with_scale=False).to(device)
    k_norm = Gemma4RMSNorm(HEAD_DIM, eps=1e-6, with_scale=False).to(device)
    rotary = get_rope(
        HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=max(token_count + 1, 16384),
        base=10000.0,
        is_neox_style=False,
        dtype=torch.float16,
    )
    q_weight = torch.full(
        (HEAD_DIM,), QUERY_SCALE, dtype=torch.float16, device=device
    ).contiguous()
    k_weight = torch.ones(HEAD_DIM, dtype=torch.float16, device=device).contiguous()

    baseline_work = torch.empty_like(source)
    fused_work = torch.empty_like(source)

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        baseline_work.copy_(source)
        q, k, _ = baseline_work.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
        q_shape, k_shape = q.shape, k.shape
        q = q_norm(q.reshape(-1, HEAD_DIM)).reshape(q_shape)
        k = k_norm(k.reshape(-1, HEAD_DIM)).reshape(k_shape)
        q = q * QUERY_SCALE
        if use_rope:
            q, k = rotary(positions, q, k)
        return q, k

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        fused_work.copy_(source)
        fused_qk_norm_rope(
            fused_work,
            Q_HEADS,
            KV_HEADS,
            KV_HEADS,
            HEAD_DIM,
            1e-6,
            q_weight,
            k_weight,
            10000.0,
            False,
            positions,
            rotary_dim=HEAD_DIM if use_rope else 0,
        )
        q, k, _ = fused_work.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
        return q, k

    q_ref, k_ref = baseline()
    q_actual, k_actual = fused()
    synchronize()
    q_metrics = error_metrics(q_actual, q_ref)
    k_metrics = error_metrics(k_actual, k_ref)

    baseline_us = median_us(lambda: baseline(), warmup, trials)
    fused_us = median_us(lambda: fused(), warmup, trials)
    return {
        "token_count": token_count,
        "mode": "rope" if use_rope else "nope",
        "q_error": q_metrics,
        "k_error": k_metrics,
        "baseline_us": baseline_us,
        "fused_us": fused_us,
        "speedup": baseline_us / fused_us,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-counts", default="1,64,1024")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    results = []
    for token_count in (int(value) for value in args.token_counts.split(",")):
        for use_rope in (True, False):
            row = run_case(token_count, use_rope, args.warmup, args.trials)
            results.append(row)
            print(json.dumps(row), flush=True)

    report = {
        "shape": {
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_SIZE,
        },
        "query_scale": QUERY_SCALE,
        "warmup": args.warmup,
        "trials": args.trials,
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
