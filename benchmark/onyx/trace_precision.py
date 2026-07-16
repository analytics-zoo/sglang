#!/usr/bin/env python3
"""Capture and compare comprehensive Onyx BF16/FP16 forward traces."""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

RAW_PARITY_INPUT_IDS = [200000, 954, 10810, 323, 4302, 373]
RAW_PARITY_FIRST_TOKEN = 328
RANK_DIR_PATTERN = re.compile(
    r"TP(?P<tp>\d+)_PP(?P<pp>\d+)_Rank(?P<rank>\d+)_pid\d+"
)


def parse_input_ids(value: str) -> list[int]:
    input_ids = [int(item) for item in value.split(",") if item]
    if not input_ids:
        raise argparse.ArgumentTypeError("At least one input token ID is required")
    return input_ids


def parse_layers(value: str) -> list[int]:
    layers = [int(item) for item in value.split(",") if item]
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return layers


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Trace output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def find_rank_dumps(root: Path, pass_index: int) -> dict[str, Path]:
    filename = f"Pass{pass_index:05d}.pt"
    rank_dumps: dict[str, Path] = {}
    for dump_path in sorted(root.glob(f"TP*_PP*_Rank*_pid*/{filename}")):
        match = RANK_DIR_PATTERN.fullmatch(dump_path.parent.name)
        if match is None:
            continue
        rank_key = (
            f"TP{match.group('tp')}_PP{match.group('pp')}_Rank{match.group('rank')}"
        )
        if rank_key in rank_dumps:
            raise RuntimeError(
                f"Multiple {filename} files found for {rank_key} under {root}"
            )
        rank_dumps[rank_key] = dump_path
    return rank_dumps


def capture_trace(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir)

    previous_mode = os.environ.get("TENSOR_DUMP_MODE")
    previous_last_token_only = os.environ.get("TENSOR_DUMP_LAST_TOKEN_ONLY")
    os.environ["TENSOR_DUMP_MODE"] = "all_io"
    if args.last_token_only:
        os.environ["TENSOR_DUMP_LAST_TOKEN_ONLY"] = "1"

    input_ids = args.input_ids
    if args.boundary_input_length is not None:
        from transformers import AutoTokenizer

        from validate_onyx import build_boundary_input

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        input_ids = build_boundary_input(tokenizer, args.boundary_input_length)

    import sglang as sgl

    engine_kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "device": "xpu",
        "tp_size": args.tp_size,
        "dtype": args.dtype,
        "load_format": args.load_format,
        "attention_backend": args.attention_backend,
        "page_size": 64,
        "mem_fraction_static": 0.95,
        "max_total_tokens": 16384,
        "swa_full_tokens_ratio": 0.25,
        "chunked_prefill_size": 1024,
        "disable_radix_cache": True,
        "max_running_requests": 1,
        "context_length": 16384,
        "disable_cuda_graph": True,
        "disable_custom_all_reduce": True,
        "disable_overlap_schedule": True,
        "skip_server_warmup": True,
        "watchdog_timeout": 300,
        "trust_remote_code": True,
        "model_impl": "sglang",
        "random_seed": 0,
        "log_level": "info",
        "debug_tensor_dump_output_folder": str(output_dir),
        "debug_tensor_dump_layers": args.dump_layers,
    }
    if args.quantization is not None:
        engine_kwargs["quantization"] = args.quantization

    engine = None
    try:
        engine = sgl.Engine(**engine_kwargs)
        result = engine.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        )
    finally:
        if engine is not None:
            engine.shutdown()
        if previous_mode is None:
            os.environ.pop("TENSOR_DUMP_MODE", None)
        else:
            os.environ["TENSOR_DUMP_MODE"] = previous_mode
        if previous_last_token_only is None:
            os.environ.pop("TENSOR_DUMP_LAST_TOKEN_ONLY", None)
        else:
            os.environ["TENSOR_DUMP_LAST_TOKEN_ONLY"] = previous_last_token_only

    rank_dumps = find_rank_dumps(output_dir, pass_index=0)
    if len(rank_dumps) != args.tp_size:
        raise RuntimeError(
            f"Expected {args.tp_size} rank dumps, found {len(rank_dumps)}: "
            f"{sorted(rank_dumps)}"
        )

    output_ids = list(result.get("output_ids", []))
    metadata = {
        "mode": "all_io",
        "model_path": args.model_path,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "load_format": args.load_format,
        "attention_backend": args.attention_backend,
        "tp_size": args.tp_size,
        "input_ids": input_ids,
        "boundary_input_length": args.boundary_input_length,
        "dump_layers": args.dump_layers,
        "last_token_only": args.last_token_only,
        "expected_first_token_id": args.expected_token_id,
        "output_ids": output_ids,
        "matches_expected_token": output_ids == [args.expected_token_id],
        "text": result.get("text"),
        "rank_dumps": {
            rank: str(path.relative_to(output_dir))
            for rank, path in rank_dumps.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def load_metadata(root: Path) -> dict[str, Any] | None:
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        return None
    return json.loads(metadata_path.read_text())


def tensor_metrics(
    reference,
    candidate,
    relative_rmse_threshold: float,
    cosine_threshold: float,
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        result.update({"shape_match": False, "diverged": True})
        return result

    result["shape_match"] = True
    result["exact_match"] = bool(torch.equal(reference, candidate))
    if reference.numel() == 0:
        result.update(
            {
                "reference_finite": True,
                "candidate_finite": True,
                "reference_rms": 0.0,
                "candidate_rms": 0.0,
                "max_abs_diff": 0.0,
                "rmse": 0.0,
                "relative_rmse": 0.0,
                "cosine_similarity": 1.0,
                "diverged": False,
            }
        )
        return result

    reference_float = reference.float()
    candidate_float = candidate.float()
    reference_finite = bool(torch.isfinite(reference_float).all())
    candidate_finite = bool(torch.isfinite(candidate_float).all())
    result["reference_finite"] = reference_finite
    result["candidate_finite"] = candidate_finite
    if not reference_finite or not candidate_finite:
        result["diverged"] = True
        return result

    difference = candidate_float - reference_float
    reference_rms = float(torch.sqrt(torch.mean(reference_float.square())))
    candidate_rms = float(torch.sqrt(torch.mean(candidate_float.square())))
    rmse = float(torch.sqrt(torch.mean(difference.square())))
    relative_rmse = rmse / max(reference_rms, 1e-12)
    max_abs_diff = float(difference.abs().max())

    reference_norm = float(torch.linalg.vector_norm(reference_float))
    candidate_norm = float(torch.linalg.vector_norm(candidate_float))
    if reference_norm == 0.0 or candidate_norm == 0.0:
        cosine_similarity = 1.0 if result["exact_match"] else 0.0
    else:
        cosine_similarity = float(
            torch.dot(reference_float.flatten(), candidate_float.flatten())
            / (reference_norm * candidate_norm)
        )

    is_floating_point = reference.is_floating_point() or candidate.is_floating_point()
    if is_floating_point:
        diverged = (
            relative_rmse > relative_rmse_threshold
            or cosine_similarity < cosine_threshold
        )
    else:
        diverged = not result["exact_match"]
    result.update(
        {
            "reference_rms": reference_rms,
            "candidate_rms": candidate_rms,
            "max_abs_diff": max_abs_diff,
            "rmse": rmse,
            "relative_rmse": relative_rmse,
            "cosine_similarity": cosine_similarity,
            "diverged": diverged,
        }
    )
    return result


def compare_traces(args: argparse.Namespace) -> None:
    import torch

    reference_dir = Path(args.reference_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    reference_dumps = find_rank_dumps(reference_dir, args.pass_index)
    candidate_dumps = find_rank_dumps(candidate_dir, args.pass_index)
    if not reference_dumps:
        raise FileNotFoundError(f"No rank dumps found under {reference_dir}")
    if reference_dumps.keys() != candidate_dumps.keys():
        raise RuntimeError(
            "Trace ranks differ: "
            f"reference={sorted(reference_dumps)}, "
            f"candidate={sorted(candidate_dumps)}"
        )

    rank_reports: dict[str, Any] = {}
    all_divergences: list[dict[str, Any]] = []
    for rank_key in reference_dumps:
        reference = torch.load(
            reference_dumps[rank_key], map_location="cpu", weights_only=False
        )
        candidate = torch.load(
            candidate_dumps[rank_key], map_location="cpu", weights_only=False
        )
        reference_keys = list(reference)
        candidate_keys = list(candidate)
        missing_in_candidate = [key for key in reference_keys if key not in candidate]
        missing_in_reference = [key for key in candidate_keys if key not in reference]

        tensors = []
        for order, name in enumerate(reference_keys):
            if name not in candidate:
                row = {
                    "order": order,
                    "name": name,
                    "missing_in_candidate": True,
                    "diverged": True,
                }
                tensors.append(row)
                all_divergences.append({"rank": rank_key, **row})
                continue
            if not isinstance(reference[name], torch.Tensor) or not isinstance(
                candidate[name], torch.Tensor
            ):
                continue
            metrics = tensor_metrics(
                reference[name],
                candidate[name],
                args.relative_rmse_threshold,
                args.cosine_threshold,
            )
            row = {"order": order, "name": name, **metrics}
            tensors.append(row)
            if row["diverged"]:
                all_divergences.append({"rank": rank_key, **row})

        first_divergence = next(
            (row for row in tensors if row["diverged"]),
            None,
        )
        rank_reports[rank_key] = {
            "reference_dump": str(reference_dumps[rank_key]),
            "candidate_dump": str(candidate_dumps[rank_key]),
            "missing_in_candidate": missing_in_candidate,
            "missing_in_reference": missing_in_reference,
            "first_divergence": first_divergence,
            "tensors": tensors,
        }

    largest_divergences = sorted(
        (
            row
            for row in all_divergences
            if row.get("relative_rmse") is not None
            and math.isfinite(row["relative_rmse"])
        ),
        key=lambda row: row["relative_rmse"],
        reverse=True,
    )[: args.top]

    report = {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "reference_metadata": load_metadata(reference_dir),
        "candidate_metadata": load_metadata(candidate_dir),
        "pass_index": args.pass_index,
        "relative_rmse_threshold": args.relative_rmse_threshold,
        "cosine_threshold": args.cosine_threshold,
        "first_divergence_by_rank": {
            rank: rank_report["first_divergence"]
            for rank, rank_report in rank_reports.items()
        },
        "largest_divergences": largest_divergences,
        "ranks": rank_reports,
    }

    if args.report is not None:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "first_divergence_by_rank": report["first_divergence_by_rank"],
        "largest_divergences": largest_divergences,
        "report": str(Path(args.report).resolve()) if args.report else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture", help="Run one request and capture every module input/output"
    )
    capture.add_argument("--model-path", default="/llm/workspace/model/onyx-hf")
    capture.add_argument("--dtype", choices=("bfloat16", "float16"), required=True)
    capture.add_argument("--quantization")
    capture.add_argument("--load-format", default="layered_fp8")
    capture.add_argument("--attention-backend", default="intel_xpu")
    capture.add_argument("--tp-size", type=int, default=2)
    capture.add_argument("--output-dir", required=True)
    capture.add_argument(
        "--input-ids",
        type=parse_input_ids,
        default=RAW_PARITY_INPUT_IDS,
    )
    capture.add_argument(
        "--expected-token-id",
        type=int,
        default=RAW_PARITY_FIRST_TOKEN,
    )
    capture.add_argument("--tokenizer-path", default="/llm/workspace/model/onyx-hf")
    capture.add_argument("--boundary-input-length", type=int)
    capture.add_argument("--dump-layers", type=parse_layers)
    capture.add_argument("--last-token-only", action="store_true")
    capture.set_defaults(func=capture_trace)

    compare = subparsers.add_parser(
        "compare", help="Compare two all-I/O trace directories"
    )
    compare.add_argument("--reference-dir", required=True)
    compare.add_argument("--candidate-dir", required=True)
    compare.add_argument("--pass-index", type=int, default=0)
    compare.add_argument("--relative-rmse-threshold", type=float, default=0.05)
    compare.add_argument("--cosine-threshold", type=float, default=0.995)
    compare.add_argument("--top", type=int, default=20)
    compare.add_argument("--report")
    compare.set_defaults(func=compare_traces)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
