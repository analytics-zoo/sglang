#!/usr/bin/env python3
"""Summarize GGUF XPU residency records from a server log.

The input is intentionally treated as plain text: no torch, XPU, or SGLang
imports are needed.  Records are emitted by ``gguf.py`` as JSON following the
``[gguf-xpu residency]`` marker.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from typing import Any, Iterable


MARKER = "[gguf-xpu residency]"
RANK_RE = re.compile(r"\bTP(\d+)\b")


def _inc(mapping: dict[str, int], key: object) -> None:
    name = str(key)
    mapping[name] = mapping.get(name, 0) + 1


def _topology(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    final = payload.get("final_rep_kinds")
    if not isinstance(final, dict):
        raise ValueError("final_rep_kinds must be an object")
    reps = final.get("reps", {})
    if not isinstance(reps, dict):
        raise ValueError("final_rep_kinds.reps must be an object")
    normalized = {
        "reps": {str(k): str(v) for k, v in sorted(reps.items(), key=lambda x: str(x[0]))},
        "merged": None if final.get("merged") is None else str(final["merged"]),
        "groups": [str(v) for v in final.get("groups", [])],
    }
    if not isinstance(final.get("groups", []), list):
        raise ValueError("final_rep_kinds.groups must be an array")
    # A compact combination is useful when comparing layers whose shard names
    # differ. Keep multiplicity for groups/reps in the full topology above.
    kinds = sorted(
        collections.Counter(
            list(normalized["reps"].values())
            + ([normalized["merged"]] if normalized["merged"] is not None else [])
            + normalized["groups"]
        ).items()
    )
    return "+".join(f"{kind}*{count}" if count != 1 else kind for kind, count in kinds), normalized


def summarize(lines: Iterable[str]) -> dict[str, Any]:
    ranks: dict[str, dict[str, Any]] = {}
    unknown_count = 0

    def rank_summary() -> dict[str, Any]:
        return {
            "record_count": 0,
            "source_enum_counts": {},
            "source_classification_counts": {},
            "source_enum_breakdown": {},
            "canonical_logical_bytes": 0,
            "final_unique_storage_bytes": 0,
            "final_unique_storage_bytes_by_canonical_kind": {},
            "final_topology_combinations": {},
            "final_topologies": {},
        }

    for line_number, line in enumerate(lines, 1):
        marker_at = line.find(MARKER)
        if marker_at < 0:
            continue
        json_text = line[marker_at + len(MARKER) :].strip()
        try:
            payload = json.loads(json_text)
            if not isinstance(payload, dict):
                raise ValueError("record JSON must be an object")
            sources = payload.get("source_descriptors", [])
            storage = payload.get("storage_keys", [])
            if not isinstance(sources, list) or not isinstance(storage, list):
                raise ValueError("source_descriptors and storage_keys must be arrays")
            combination, topology = _topology(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: cannot parse residency record: {exc}") from exc

        rank_matches = RANK_RE.findall(line[:marker_at])
        if not rank_matches:
            rank = "unknown"
            unknown_count += 1
        else:
            rank = rank_matches[-1]
        summary = ranks.setdefault(rank, rank_summary())
        summary["record_count"] += 1
        _inc(summary["final_topology_combinations"], combination)
        topology_key = json.dumps(topology, sort_keys=True, separators=(",", ":"))
        _inc(summary["final_topologies"], topology_key)

        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(f"line {line_number}: source descriptor must be an object")
            source_type = str(source.get("source_type", "unknown"))
            classification = str(source.get("classification", "unknown"))
            _inc(summary["source_enum_counts"], source_type)
            _inc(summary["source_classification_counts"], classification)
            try:
                logical_bytes = int(source.get("logical_tensor_bytes", 0))
                canonical_bytes = int(source.get("canonical_logical_bytes", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: invalid source byte count") from exc
            summary["canonical_logical_bytes"] += canonical_bytes
            breakdown = summary["source_enum_breakdown"].setdefault(
                source_type,
                {
                    "count": 0,
                    "classification_counts": {},
                    "logical_tensor_bytes": 0,
                    "canonical_logical_bytes": 0,
                },
            )
            breakdown["count"] += 1
            _inc(breakdown["classification_counts"], classification)
            breakdown["logical_tensor_bytes"] += logical_bytes
            breakdown["canonical_logical_bytes"] += canonical_bytes

        # Keep one canonical record for each (device, pointer, size) key over
        # the complete rank, while allowing a key to be referenced by aliases
        # with more than one final kind.
        seen = summary.setdefault("_storage", {})
        for record in storage:
            if not isinstance(record, dict) or not isinstance(record.get("storage_key"), dict):
                raise ValueError(f"line {line_number}: invalid storage key record")
            key_data = record["storage_key"]
            try:
                key = (str(key_data["device"]), int(key_data["data_ptr"]), int(key_data["nbytes"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: malformed storage_key") from exc
            kinds = record.get("kinds", [])
            if not isinstance(kinds, list):
                raise ValueError(f"line {line_number}: storage kinds must be an array")
            seen.setdefault(key, set()).update(str(kind) for kind in kinds)

    if not ranks:
        raise ValueError("no residency records found")
    if unknown_count:
        print(f"warning: {unknown_count} residency record(s) had unknown TP rank", file=sys.stderr)
    for summary in ranks.values():
        storage = summary.pop("_storage", {})
        summary["final_unique_storage_bytes"] = sum(key[2] for key in storage)
        by_kind: dict[str, int] = {}
        for key, kinds in storage.items():
            for kind in kinds:
                by_kind[kind] = by_kind.get(kind, 0) + key[2]
        summary["final_unique_storage_bytes_by_canonical_kind"] = dict(sorted(by_kind.items()))
    return {"schema": "gguf_xpu_residency_summary_v1", "ranks": dict(sorted(ranks.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", default="-", help="log file, or - for stdin")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = parser.parse_args()
    try:
        if args.log == "-":
            result = summarize(sys.stdin)
        else:
            with open(args.log, encoding="utf-8", errors="replace") as stream:
                result = summarize(stream)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True, separators=None if args.pretty else (",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
