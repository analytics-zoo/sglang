#!/usr/bin/env python3
"""Measure bsz=1 Onyx TTFT and TPOT with real tokenized text."""

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from validate_onyx import BOUNDARY_SEED


def build_input(tokenizer: Any, input_len: int) -> list[int]:
    seed_ids = tokenizer.encode(BOUNDARY_SEED, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("Onyx benchmark seed encoded to zero tokens")
    prefix = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    repeats = math.ceil((input_len - len(prefix)) / len(seed_ids))
    input_ids = (prefix + seed_ids * repeats)[:input_len]
    if len(input_ids) != input_len:
        raise AssertionError(f"Expected {input_len} input tokens, got {len(input_ids)}")
    return input_ids


def generate(
    base_url: str,
    input_ids: list[int],
    output_len: int,
    timeout: float,
) -> dict[str, float | int]:
    body = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    events = 0
    last_payload = ""
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            last_payload = payload
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            events += 1

    end = time.perf_counter()
    if first_token_time is None or last_token_time is None:
        raise RuntimeError("Streaming request returned no token events")
    if events != output_len:
        raise RuntimeError(
            f"Expected {output_len} stream events, got {events}; "
            f"last payload: {last_payload[:500]}"
        )

    ttft = first_token_time - start
    decode_time = last_token_time - first_token_time
    tpot = decode_time / (events - 1) if events > 1 else float("nan")
    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": end - start,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31888")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--input-lengths",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[1024, 2048, 4096, 8192, 16000],
    )
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    report: dict[str, Any] = {
        "label": args.label,
        "base_url": args.base_url,
        "input_source": "repeated-real-tokenized-onyx-boundary-text",
        "output_length": args.output_length,
        "warmup": args.warmup,
        "trials": args.trials,
        "loadavg_before": list(os.getloadavg()),
        "results": [],
    }

    for input_len in args.input_lengths:
        input_ids = build_input(tokenizer, input_len)
        for _ in range(args.warmup):
            generate(args.base_url, input_ids, args.output_length, args.timeout)
        trials = [
            generate(args.base_url, input_ids, args.output_length, args.timeout)
            for _ in range(args.trials)
        ]
        ttft_ms = statistics.median(row["ttft_s"] for row in trials) * 1000
        tpot_ms = statistics.median(row["tpot_s"] for row in trials) * 1000
        e2e_s = statistics.median(row["e2e_s"] for row in trials)
        row = {
            "input_length": input_len,
            "ttft_ms_median": ttft_ms,
            "tpot_ms_median": tpot_ms,
            "decode_tokens_per_s": 1000 / tpot_ms,
            "e2e_s_median": e2e_s,
            "trials": trials,
        }
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    report["loadavg_after"] = list(os.getloadavg())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
