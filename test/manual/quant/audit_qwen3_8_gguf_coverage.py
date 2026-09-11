#!/usr/bin/env python3
"""Audit source and XPU native/fallback coverage of a Qwen3.8 GGUF.

This is a bounded metadata audit, not a model loader.  It materializes only
the first row of each 2-D GGUF tensor, asks the real XPU preparation helper for
the resident representation, and extrapolates its byte footprint to all rows.
The extrapolated values are logical resident bytes; allocator usage is not
measured here.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any

import gguf
import torch


TARGET_TYPES = {"IQ4_NL", "IQ4_XS", "Q3_K", "IQ3_S"}


def _rep_bytes(value: Any) -> int:
    """Count tensor payload bytes in a resident representation tuple."""
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_rep_bytes(item) for item in value)
    return 0


def _kind(rep: Any) -> str:
    return str(rep[0]) if isinstance(rep, (tuple, list)) and rep else "unknown"


def _group(name: str) -> str:
    return "mtp_blk.64" if name.startswith("blk.64.") else "main"


def _source_bytes(data: Any) -> int:
    # GGUFReader tensors expose numpy arrays.  nbytes avoids constructing a
    # torch tensor for the full checkpoint.
    return int(getattr(data, "nbytes", 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="permit fallback for IQ4_NL/IQ4_XS/Q3_K/IQ3_S (old wheel check)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    # Import after argument parsing so --help works without an XPU extension.
    from sglang.srt.layers.quantization.gguf import _xpu_prepare_shard

    reader = gguf.GGUFReader(args.gguf_path)
    report: dict[str, Any] = {
        "gguf_path": os.path.abspath(args.gguf_path),
        "logical_bytes_definition": (
            "resident representation bytes extrapolated from one row; "
            "does not apply TP sharding and does not include merge duplicates; "
            "does not measure allocator bytes"
        ),
        "groups": {},
        "totals": {"source_tensors": 0, "source_bytes": 0},
    }
    failures: list[str] = []

    for tensor in reader.tensors:
        name = tensor.name
        group = _group(name)
        type_name = tensor.tensor_type.name
        data = tensor.data
        source_bytes = _source_bytes(data)
        bucket = report["groups"].setdefault(
            group,
            {
                "source_tensors": 0,
                "source_bytes": 0,
                "types": {},
                "non_matrix": [],
            },
        )
        entry = bucket["types"].setdefault(
            type_name,
            {
                "source_tensors": 0,
                "source_bytes": 0,
                "native_tensors": 0,
                "fallback_tensors": 0,
                "native_logical_bytes": 0,
                "fallback_logical_bytes": 0,
                "unquantized_tensors": 0,
                "unquantized_logical_bytes": 0,
                "resident_kinds": collections.Counter(),
            },
        )
        bucket["source_tensors"] += 1
        bucket["source_bytes"] += source_bytes
        report["totals"]["source_tensors"] += 1
        report["totals"]["source_bytes"] += source_bytes
        entry["source_tensors"] += 1
        entry["source_bytes"] += source_bytes

        # F32 vectors (norms, biases, etc.) are intentionally reported as raw
        # unquantized metadata.  The helper expects a matrix for row reps.
        if type_name == "F32":
            # F32 is intentionally not classified as quantized fallback.  The
            # XPU helper converts matrix weights to the network dtype, but the
            # report keeps this natural/unquantized population separate.
            rows = int(data.shape[0]) if getattr(data, "ndim", 0) >= 2 else 1
            cols = int(data.size // rows) if rows else 0
            item_bytes = rows * cols * 4
            entry["unquantized_tensors"] += 1
            entry["unquantized_logical_bytes"] += item_bytes
            if getattr(data, "ndim", 0) < 2:
                bucket["non_matrix"].append({"name": name, "source_bytes": source_bytes})
            continue
        if getattr(data, "ndim", 0) < 2:
            bucket["non_matrix"].append(
                {"name": name, "type": type_name, "source_bytes": source_bytes}
            )
            continue

        rows = int(data.shape[0])
        # Keep one row only.  torch.tensor copies that row and releases it at
        # the end of this iteration; no full GGUF tensor is retained.
        row = torch.tensor(data[:1])
        try:
            rep = _xpu_prepare_shard(row, int(tensor.tensor_type), torch.float16)
            kind = _kind(rep)
            logical_bytes = _rep_bytes(rep) * rows
        except Exception as exc:  # report unsupported wheel/kernel clearly
            kind = "error"
            logical_bytes = 0
            failures.append(f"{name}: {type_name}: {type(exc).__name__}: {exc}")

        entry["resident_kinds"][kind] += 1
        if kind == "fp16":
            entry["fallback_tensors"] += 1
            entry["fallback_logical_bytes"] += logical_bytes
        elif kind != "error":
            entry["native_tensors"] += 1
            entry["native_logical_bytes"] += logical_bytes

    # Counter is convenient while accumulating but should be plain JSON.
    for group in report["groups"].values():
        for entry in group["types"].values():
            entry["resident_kinds"] = dict(entry["resident_kinds"])

    watched_fallbacks = []
    main_types = report["groups"].get("main", {}).get("types", {})
    for type_name in sorted(TARGET_TYPES):
        item = main_types.get(type_name, {})
        count = int(item.get("fallback_tensors", 0))
        if count:
            watched_fallbacks.append({"type": type_name, "fallback_tensors": count})
    report["watched_main_fallbacks"] = watched_fallbacks
    report["watched_main_presence"] = {
        type_name: int(main_types.get(type_name, {}).get("source_tensors", 0))
        for type_name in sorted(TARGET_TYPES)
    }
    report["probe_failures"] = failures

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GGUF: {report['gguf_path']}")
        print(
            "logical bytes are extrapolated resident rep bytes; "
            "allocator usage is not measured; TP sharding and merge duplicates "
            "are excluded"
        )
        for group_name, group in report["groups"].items():
            print(
                f"[{group_name}] source_tensors={group['source_tensors']} "
                f"source_bytes={group['source_bytes']}"
            )
            for type_name, item in sorted(group["types"].items()):
                print(
                    f"  {type_name}: source={item['source_tensors']} "
                    f"native={item['native_tensors']} "
                    f"fallback_fp16={item['fallback_tensors']} "
                    f"native_bytes={item['native_logical_bytes']} "
                    f"fallback_bytes={item['fallback_logical_bytes']} "
                    f"unquantized={item['unquantized_tensors']} "
                    f"unquantized_bytes={item['unquantized_logical_bytes']} "
                    f"kinds={dict(item['resident_kinds'])}"
                )
        if failures:
            print("probe_failures:")
            print("\n".join(f"  {failure}" for failure in failures))

    if failures:
        return 3
    if watched_fallbacks and not args.allow_fallback:
        print(
            "ERROR: watched main matrix types still use FP16 fallback; "
            "rerun with --allow-fallback only for an old-wheel check.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
