#!/usr/bin/env python3
"""Measure Onyx long-context multi-needle retrieval through an SGLang server."""

from __future__ import annotations

import argparse
import json
import random
import string
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_LENGTHS = (16384, 18432, 20480, 24576, 28672, 32768)
DEFAULT_NEEDLE_RATIOS = (0.05, 0.25, 0.50, 0.75, 0.95)
CONTENT_MARKER = "ONYX_LONG_CONTEXT_CONTENT_MARKER_7B4A91D3"


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def comma_ratios(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("ratios must be strictly between 0 and 1")
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("ratios must be unique and increasing")
    return values


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def repeated_filler(pattern: list[int], length: int) -> list[int]:
    if length < 0:
        raise ValueError(f"negative filler length: {length}")
    quotient, remainder = divmod(length, len(pattern))
    return pattern * quotient + pattern[:remainder]


def nonce(rng: random.Random) -> str:
    alphabet = string.ascii_uppercase + string.digits
    raw = "".join(rng.choice(alphabet) for _ in range(12))
    return f"{raw[:4]}-{raw[4:8]}-{raw[8:]}"


def chat_scaffold(tokenizer: Any) -> tuple[list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a precise information-retrieval assistant.",
            },
            {"role": "user", "content": CONTENT_MARKER},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    marker_start = rendered.find(CONTENT_MARKER)
    if marker_start < 0:
        raise RuntimeError("chat template did not preserve the content marker")
    marker_end = marker_start + len(CONTENT_MARKER)
    return encode(tokenizer, rendered[:marker_start]), encode(
        tokenizer, rendered[marker_end:]
    )


def build_case(
    tokenizer: Any,
    prompt_length: int,
    ratios: list[float],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random((prompt_length << 16) ^ seed)
    prefix_ids, suffix_ids = chat_scaffold(tokenizer)
    instruction_ids = encode(
        tokenizer,
        "Read the archive. Memorize every CRITICAL_RECORD exactly and ignore all "
        "neutral filler records.\n",
    )
    filler_pattern = encode(
        tokenizer,
        " Neutral archive filler record: cedar harbor, no critical information.\n",
    )
    if not filler_pattern:
        raise RuntimeError("filler text encoded to zero tokens")

    expected: dict[str, str] = {}
    needle_ids: list[tuple[str, list[int]]] = []
    for ratio in ratios:
        key = f"P{round(ratio * 100):02d}"
        value = nonce(rng)
        expected[key] = value
        needle_ids.append(
            (key, encode(tokenizer, f"\nCRITICAL_RECORD {key}={value}\n"))
        )

    keys = ",".join(expected)
    query_ids = encode(
        tokenizer,
        "\nEnd of archive. Return exactly one compact JSON object and nothing else. "
        f"The keys must be {keys}; each value must exactly match its CRITICAL_RECORD.\n",
    )

    minimum_length = (
        len(prefix_ids)
        + len(instruction_ids)
        + sum(len(ids) for _, ids in needle_ids)
        + len(query_ids)
        + len(suffix_ids)
    )
    if prompt_length < minimum_length + 128:
        raise ValueError(
            f"prompt length {prompt_length} is too short; need at least {minimum_length + 128}"
        )

    input_ids = [*prefix_ids, *instruction_ids]
    positions: dict[str, int] = {}
    for ratio, (key, ids) in zip(ratios, needle_ids):
        target_position = round((prompt_length - 1) * ratio)
        if target_position < len(input_ids):
            raise ValueError(
                f"needle {key} target {target_position} overlaps the prompt prefix"
            )
        input_ids.extend(
            repeated_filler(filler_pattern, target_position - len(input_ids))
        )
        positions[key] = len(input_ids)
        input_ids.extend(ids)

    tail_length = prompt_length - len(input_ids) - len(query_ids) - len(suffix_ids)
    input_ids.extend(repeated_filler(filler_pattern, tail_length))
    query_position = len(input_ids)
    input_ids.extend(query_ids)
    input_ids.extend(suffix_ids)
    if len(input_ids) != prompt_length:
        raise AssertionError(f"built {len(input_ids)} tokens, expected {prompt_length}")

    return {
        "input_ids": input_ids,
        "expected": expected,
        "positions": positions,
        "query_position": query_position,
        "distances": {
            key: query_position - position for key, position in positions.items()
        },
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:2000]}") from error
    result = json.loads(body)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"expected JSON object response, got {type(result).__name__}"
        )
    return result


def parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def write_report(report: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31888")
    parser.add_argument(
        "--model-path", type=Path, default=Path("/llm/workspace/model/onyx-hf")
    )
    parser.add_argument(
        "--prompt-lengths",
        type=comma_ints,
        default=list(DEFAULT_PROMPT_LENGTHS),
        help="comma-separated exact input-token lengths",
    )
    parser.add_argument(
        "--needle-ratios",
        type=comma_ratios,
        default=list(DEFAULT_NEEDLE_RATIOS),
    )
    parser.add_argument(
        "--seeds", type=comma_ints, default=[1], help="comma-separated nonzero seeds"
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    declared_tokenizer_limit = int(tokenizer.model_max_length)
    # This suppresses tokenizer-only warnings while constructing extrapolation
    # inputs. It does not modify the server model configuration.
    tokenizer.model_max_length = 10**9

    report: dict[str, Any] = {
        "test": "sglang_onyx_long_context_multi_needle",
        "base_url": args.base_url,
        "model_path": str(args.model_path),
        "tokenizer_declared_model_max_length": declared_tokenizer_limit,
        "prompt_lengths": args.prompt_lengths,
        "needle_ratios": args.needle_ratios,
        "seeds": args.seeds,
        "max_new_tokens": args.max_new_tokens,
        "dry_run": args.dry_run,
        "runs": [],
    }

    score_rows: dict[str, list[bool]] = defaultdict(list)
    score_lengths: dict[int, list[bool]] = defaultdict(list)
    for prompt_length in args.prompt_lengths:
        for seed in args.seeds:
            case = build_case(tokenizer, prompt_length, args.needle_ratios, seed)
            row: dict[str, Any] = {
                "prompt_tokens": len(case["input_ids"]),
                "requested_total_tokens": len(case["input_ids"]) + args.max_new_tokens,
                "seed": seed,
                "positions": case["positions"],
                "query_position": case["query_position"],
                "distances": case["distances"],
                "expected": case["expected"],
            }
            if args.dry_run:
                row["phase"] = "dry_run"
                report["runs"].append(row)
                continue

            start = time.perf_counter()
            try:
                result = post_json(
                    f"{args.base_url.rstrip('/')}/generate",
                    {
                        "input_ids": case["input_ids"],
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": args.max_new_tokens,
                        },
                    },
                    args.timeout,
                )
                response_text = result.get("text")
                if not isinstance(response_text, str):
                    raise RuntimeError(
                        "response does not contain a string 'text' field"
                    )
                parsed = parse_json_object(response_text)
                exact = {
                    key: parsed is not None and parsed.get(key) == value
                    for key, value in case["expected"].items()
                }
                for key, passed in exact.items():
                    score_rows[key].append(passed)
                    score_lengths[prompt_length].append(passed)
                row.update(
                    {
                        "phase": "complete",
                        "elapsed_seconds": time.perf_counter() - start,
                        "response_text": response_text,
                        "parsed_json": parsed,
                        "exact": exact,
                        "correct": sum(exact.values()),
                        "total": len(exact),
                        "accuracy": sum(exact.values()) / len(exact),
                        "meta_info": result.get("meta_info"),
                    }
                )
            except Exception as error:
                row.update(
                    {
                        "phase": "error",
                        "elapsed_seconds": time.perf_counter() - start,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "correct": 0,
                        "total": len(case["expected"]),
                        "accuracy": 0.0,
                    }
                )
                for key in case["expected"]:
                    score_rows[key].append(False)
                    score_lengths[prompt_length].append(False)
            report["runs"].append(row)

    if args.dry_run:
        report.update({"passed": True, "phase": "dry_run"})
    else:
        total = sum(row["total"] for row in report["runs"])
        correct = sum(row["correct"] for row in report["runs"])
        report.update(
            {
                "phase": "complete",
                "correct": correct,
                "total": total,
                "overall_accuracy": correct / total if total else 0.0,
                "accuracy_by_position": {
                    key: sum(values) / len(values) for key, values in score_rows.items()
                },
                "accuracy_by_prompt_length": {
                    str(length): sum(values) / len(values)
                    for length, values in score_lengths.items()
                },
                "passed": correct == total and total > 0,
            }
        )

    write_report(report, args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
