#!/usr/bin/env python3
"""Qualify Onyx TP=2 online-FP8 linear shapes on Intel XPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.srt.layers.quantization import fp8_utils


SHAPES = (
    ("qkv_proj", 6656, 2304),
    ("output_gate_proj", 6656, 2048),
    ("o_proj", 2048, 6656),
    ("gate_up_proj", 6656, 19968),
    ("down_proj", 9984, 6656),
)
SMALL_M = (1, 2, 4, 8, 16, 32, 64)
PREFILL_M = (65, 128, 256, 1024, 2048, 4096, 8192, 16384)


def synchronize() -> None:
    torch.xpu.synchronize()


def timed_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        result = function()
        del result
    synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        result = function()
        del result
    synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def iteration_count(m: int) -> int:
    if m <= 64:
        return 20
    if m <= 1024:
        return 10
    if m <= 4096:
        return 5
    return 3


def make_weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randn(n, k, device="xpu", dtype=torch.float16) * 0.02
    weight_nk, weight_scale = fp8_utils.input_to_float8(source)
    del source
    return weight_nk.t(), weight_scale.reshape(1)


def dequantized_reference(
    input_tensor: torch.Tensor,
    weight_kn: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    weight_nk = weight_kn.t()
    dequantized = weight_nk.to(torch.float16) * weight_scale.to(torch.float16)
    return torch.matmul(input_tensor, dequantized.t())


def fast_path(
    input_tensor: torch.Tensor,
    weight_kn: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if input_tensor.shape[0] > 64:
        return fp8_utils._fp8_gemm_w8a16(
            input_tensor, weight_kn, weight_scale, None
        )

    weight_nk = getattr(weight_kn, "_esimd_t", None)
    if weight_nk is None:
        weight_nk = weight_kn.t().contiguous()
        weight_kn._esimd_t = weight_nk
    scale_1d = weight_scale.to(torch.float32).reshape(-1)[:1].contiguous()
    output = torch.empty(
        input_tensor.shape[0],
        weight_nk.shape[0],
        dtype=torch.float16,
        device=input_tensor.device,
    )
    fp8_utils._esimd_gemm_fp8_pert(input_tensor, weight_nk, scale_1d, output)
    return output


def fallback_path(
    input_tensor: torch.Tensor,
    weight_kn: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    saved_esimd = fp8_utils._esimd_gemm_fp8_pert
    saved_w8a16 = fp8_utils._fp8_gemm_w8a16
    try:
        if input_tensor.shape[0] <= 64:
            fp8_utils._esimd_gemm_fp8_pert = None
        else:
            fp8_utils._fp8_gemm_w8a16 = None
        return fp8_utils.apply_fp8_linear(
            input_tensor,
            weight_kn,
            weight_scale,
            cutlass_fp8_supported=False,
            pad_output=False,
        )
    finally:
        fp8_utils._esimd_gemm_fp8_pert = saved_esimd
        fp8_utils._fp8_gemm_w8a16 = saved_w8a16


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    relative_difference = difference.abs() / reference_float.abs().clamp_min(1e-6)
    reference_rms = torch.sqrt(torch.mean(reference_float.square()))
    rmse = torch.sqrt(torch.mean(difference.square()))
    relative_rmse = rmse / reference_rms.clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        reference_float.flatten(), candidate_float.flatten(), dim=0
    )
    return {
        "finite": bool(torch.isfinite(candidate_float).all()),
        "max_abs_diff": float(difference.abs().max()),
        "max_relative_error": float(relative_difference.max()),
        "relative_rmse": float(relative_rmse),
        "cosine_similarity": float(cosine),
    }


def run_case(
    name: str,
    k: int,
    n: int,
    m: int,
    weight_kn: torch.Tensor,
    weight_scale: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict:
    torch.manual_seed(m)
    input_tensor = torch.randn(m, k, device="xpu", dtype=torch.float16) * 0.02
    expected_path = "esimd_small_m" if m <= 64 else "packaged_onednn_w8a16"

    candidate = fast_path(input_tensor, weight_kn, weight_scale)
    repeated_candidate = fast_path(input_tensor, weight_kn, weight_scale)
    reference = dequantized_reference(input_tensor, weight_kn, weight_scale)
    metrics = tensor_metrics(reference, candidate)
    repeat_metrics = tensor_metrics(candidate, repeated_candidate)
    bitwise_repeatable = torch.equal(candidate, repeated_candidate)
    repeatable = (
        repeat_metrics["finite"]
        and repeat_metrics["relative_rmse"] <= 0.001
        and repeat_metrics["cosine_similarity"] >= 0.999999
    )
    del candidate, repeated_candidate, reference

    iterations = iteration_count(m)
    fast_samples = []
    fallback_samples = []
    for repeat in range(repeats):
        measurements = (
            (
                ("fast", fast_path),
                ("fallback", fallback_path),
            )
            if repeat % 2 == 0
            else (
                ("fallback", fallback_path),
                ("fast", fast_path),
            )
        )
        for label, function in measurements:
            duration = timed_ms(
                lambda function=function: function(
                    input_tensor, weight_kn, weight_scale
                ),
                warmup,
                iterations,
            )
            (fast_samples if label == "fast" else fallback_samples).append(duration)
    fast_ms = statistics.median(fast_samples)
    fallback_ms = statistics.median(fallback_samples)
    row = {
        "module": name,
        "m": m,
        "k": k,
        "n": n,
        "path": expected_path,
        "fast_ms": fast_ms,
        "fallback_ms": fallback_ms,
        "fast_samples_ms": fast_samples,
        "fallback_samples_ms": fallback_samples,
        "speedup": fallback_ms / fast_ms,
        "repeatable": repeatable,
        "bitwise_repeatable": bitwise_repeatable,
        "repeat_relative_rmse": repeat_metrics["relative_rmse"],
        "repeat_cosine_similarity": repeat_metrics["cosine_similarity"],
        "repeat_max_abs_diff": repeat_metrics["max_abs_diff"],
        **metrics,
    }
    row["numerically_qualified"] = (
        row["finite"]
        and row["repeatable"]
        and row["relative_rmse"] <= 0.05
        and row["cosine_similarity"] >= 0.999
    )
    row["performance_qualified"] = fast_ms < fallback_ms
    del input_tensor
    torch.xpu.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--small-only", action="store_true")
    group.add_argument("--prefill-only", action="store_true")
    parser.add_argument(
        "--module",
        action="append",
        choices=[shape[0] for shape in SHAPES],
        help="Only benchmark the named module shape; may be repeated.",
    )
    parser.add_argument(
        "--m",
        action="append",
        type=int,
        help="Only benchmark this token count; may be repeated.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("This benchmark requires an Intel XPU")
    if fp8_utils._esimd_gemm_fp8_pert is None:
        raise RuntimeError("Packaged ESIMD FP8 small-M op is unavailable")
    if fp8_utils._fp8_gemm_w8a16 is None:
        raise RuntimeError("Packaged oneDNN W8A16 op is unavailable")

    m_values = (
        SMALL_M
        if args.small_only
        else PREFILL_M
        if args.prefill_only
        else SMALL_M + PREFILL_M
    )
    if args.m:
        invalid_m = sorted(set(args.m) - set(m_values))
        if invalid_m:
            raise ValueError(
                f"M values {invalid_m} are outside the selected benchmark matrix"
            )
        m_values = tuple(dict.fromkeys(args.m))
    shapes = (
        tuple(shape for shape in SHAPES if shape[0] in args.module)
        if args.module
        else SHAPES
    )
    rows = []
    for name, k, n in shapes:
        weight_kn, weight_scale = make_weight(n, k)
        for m in m_values:
            row = run_case(
                name,
                k,
                n,
                m,
                weight_kn,
                weight_scale,
                args.warmup,
                args.repeats,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del weight_kn, weight_scale
        torch.xpu.empty_cache()

    report = {
        "device": torch.xpu.get_device_name(0),
        "rows": rows,
        "all_numerically_qualified": all(
            row["numerically_qualified"] for row in rows
        ),
        "all_performance_qualified": all(
            row["performance_qualified"] for row in rows
        ),
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
