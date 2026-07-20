#!/usr/bin/env python3
"""Benchmark Onyx packed-QKV KV-cache scatter alternatives on XPU."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from custom_esimd_kernels_sglang import esimd_kv_scatter


QKV_WIDTH = 2304
Q_WIDTH = 2048
ROW_DIM = 128


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


def run_case(
    token_count: int,
    warmup: int,
    trials: int,
    include_strided_kernel: bool,
) -> dict:
    device = torch.device("xpu")
    torch.manual_seed(0)
    packed = torch.randn(
        token_count, QKV_WIDTH, dtype=torch.float16, device=device
    ).contiguous()
    k = packed[:, Q_WIDTH : Q_WIDTH + ROW_DIM]
    v = packed[:, Q_WIDTH + ROW_DIM : Q_WIDTH + 2 * ROW_DIM]
    slots = max(token_count * 2, 16384)
    indices = torch.arange(token_count, dtype=torch.int64, device=device) * 2
    k_cache = torch.zeros(slots, ROW_DIM, dtype=torch.float16, device=device)
    v_cache = torch.zeros_like(k_cache)

    def native() -> None:
        k_cache[indices] = k
        v_cache[indices] = v

    def contiguous_fused() -> None:
        esimd_kv_scatter(
            k.contiguous(), v.contiguous(), k_cache, v_cache, indices
        )

    native()
    synchronize()
    native_k = k_cache[indices].clone()
    native_v = v_cache[indices].clone()
    k_cache.zero_()
    v_cache.zero_()
    contiguous_fused()
    synchronize()
    contiguous_correct = bool(
        torch.equal(k_cache[indices], native_k)
        and torch.equal(v_cache[indices], native_v)
    )

    timings = {
        "native_us": median_us(native, warmup, trials),
        "contiguous_copy_fused_us": median_us(
            contiguous_fused, warmup, trials
        ),
    }
    result = {
        "token_count": token_count,
        "source_stride": list(k.stride()),
        "contiguous_copy_fused_correct": contiguous_correct,
        **timings,
    }

    if include_strided_kernel:
        def strided_fused() -> None:
            esimd_kv_scatter(k, v, k_cache, v_cache, indices)

        k_cache.zero_()
        v_cache.zero_()
        strided_fused()
        synchronize()
        result["strided_fused_correct"] = bool(
            torch.equal(k_cache[indices], native_k)
            and torch.equal(v_cache[indices], native_v)
        )
        result["strided_fused_us"] = median_us(
            strided_fused, warmup, trials
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-counts", default="1,64,1024")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--include-strided-kernel", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    results = []
    for token_count in (int(value) for value in args.token_counts.split(",")):
        row = run_case(
            token_count,
            args.warmup,
            args.trials,
            args.include_strided_kernel,
        )
        results.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "qkv_width": QKV_WIDTH,
        "row_dim": ROW_DIM,
        "dtype": "float16",
        "warmup": args.warmup,
        "trials": args.trials,
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
