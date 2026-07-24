#!/usr/bin/env python3
"""Check Onyx numerical correctness across online-serving concurrency levels.

The prompt construction and answer extraction intentionally match
``sglang.test.simple_eval_gsm8k``: the first five GSM8K test examples are used
as few-shot demonstrations, evaluation starts after those examples, and the
last number in each response is compared with the last number in the gold
answer.  Unlike the generic evaluator, this script retains every response and
compares each concurrency level with the concurrency=1 prediction baseline.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
INVALID = -9999999


def get_answer_value(answer: str) -> int | float:
    numbers = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def get_one_example(row: dict[str, str], include_answer: bool) -> str:
    text = f"Question: {row['question']}\nAnswer:"
    if include_answer:
        text += f" {row['answer']}"
    return text


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GSM8K_URL, path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def flush_cache(base_url: str, timeout: float) -> None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/flush_cache", data=b"", method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"flush_cache returned HTTP {response.status}")


def ask(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    latency = time.perf_counter() - start
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    return {
        "response": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "latency_s": latency,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31888")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", default="/tmp/gsm8k_test.jsonl")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument(
        "--eval-indices",
        type=lambda value: [int(item) for item in value.split(",")],
        help="Optional zero-based indices in the post-few-shot evaluation split.",
    )
    parser.add_argument(
        "--concurrency",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[1, 2, 3, 4],
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.num_examples <= 0 or args.num_shots < 0:
        raise ValueError("num-examples must be positive and num-shots non-negative")
    if not args.concurrency or any(value <= 0 for value in args.concurrency):
        raise ValueError("all concurrency levels must be positive")
    if args.concurrency[0] != 1:
        raise ValueError("the first concurrency level must be 1 for baseline comparison")

    data_path = Path(args.data_path)
    rows = load_rows(data_path)
    source_indices = (
        args.eval_indices
        if args.eval_indices is not None
        else list(range(args.num_examples))
    )
    if not source_indices or any(index < 0 for index in source_indices):
        raise ValueError("eval-indices must contain non-negative indices")
    required = args.num_shots + max(source_indices) + 1
    if len(rows) < required:
        raise ValueError(f"dataset has {len(rows)} rows; {required} are required")

    few_shot = "".join(
        get_one_example(row, include_answer=True) + "\n\n"
        for row in rows[: args.num_shots]
    )
    evaluation_split = rows[args.num_shots :]
    eval_rows = [evaluation_split[index] for index in source_indices]
    args.num_examples = len(eval_rows)
    prompts = [few_shot + get_one_example(row, include_answer=False) for row in eval_rows]
    golds = [get_answer_value(row["answer"]) for row in eval_rows]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "data_path": str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "num_examples": args.num_examples,
        "eval_indices": source_indices,
        "num_shots": args.num_shots,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "concurrency_levels": args.concurrency,
        "runs": [],
    }
    baseline_predictions: list[int | float] | None = None
    baseline_correct: list[bool] | None = None

    for concurrency in args.concurrency:
        flush_cache(args.base_url, args.timeout)
        start = time.perf_counter()

        def work(index: int) -> dict[str, Any]:
            try:
                result = ask(
                    args.base_url,
                    args.model,
                    prompts[index],
                    args.max_tokens,
                    args.timeout,
                )
                result["error"] = None
            except Exception as error:
                result = {
                    "response": "",
                    "finish_reason": None,
                    "latency_s": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            prediction = get_answer_value(result["response"])
            result.update(
                {
                    "index": index,
                    "source_index": source_indices[index],
                    "question": eval_rows[index]["question"],
                    "gold": golds[index],
                    "prediction": prediction,
                    "correct": prediction == golds[index],
                }
            )
            return result

        details: list[dict[str, Any] | None] = [None] * args.num_examples
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(work, index): index
                for index in range(args.num_examples)
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                details[futures[future]] = future.result()
                if completed % 10 == 0 or completed == args.num_examples:
                    print(
                        f"concurrency={concurrency} progress="
                        f"{completed}/{args.num_examples}",
                        flush=True,
                    )

        resolved_details = [row for row in details if row is not None]
        if len(resolved_details) != args.num_examples:
            raise RuntimeError("some GSM8K worker results are missing")
        details = resolved_details

        wall_time = time.perf_counter() - start
        predictions = [row["prediction"] for row in details]
        correct = [bool(row["correct"]) for row in details]
        latencies = [
            float(row["latency_s"])
            for row in details
            if row["latency_s"] is not None
        ]
        completion_tokens = sum(
            int(row["completion_tokens"] or 0) for row in details
        )
        errors = sum(row["error"] is not None for row in details)
        invalid = sum(row["prediction"] == INVALID for row in details)

        if baseline_predictions is None:
            baseline_predictions = predictions
            baseline_correct = correct
            changed = regressions = improvements = 0
        else:
            assert baseline_correct is not None
            changed = sum(
                prediction != baseline
                for prediction, baseline in zip(predictions, baseline_predictions)
            )
            regressions = sum(
                was_correct and not is_correct
                for was_correct, is_correct in zip(baseline_correct, correct)
            )
            improvements = sum(
                not was_correct and is_correct
                for was_correct, is_correct in zip(baseline_correct, correct)
            )

        summary = {
            "concurrency": concurrency,
            "accuracy": sum(correct) / args.num_examples,
            "correct": sum(correct),
            "invalid": invalid,
            "errors": errors,
            "wall_time_s": wall_time,
            "output_throughput_tokens_s": (
                completion_tokens / wall_time if wall_time > 0 else None
            ),
            "request_latency_s_median": statistics.median(latencies),
            "request_latency_s_p95": percentile(latencies, 0.95),
            "prediction_changes_vs_concurrency_1": changed,
            "regressions_vs_concurrency_1": regressions,
            "improvements_vs_concurrency_1": improvements,
        }
        report["runs"].append({"summary": summary, "details": details})
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
