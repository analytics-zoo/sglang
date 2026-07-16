#!/usr/bin/env python3
"""Evaluate Onyx on zero-shot ARC-Challenge through the Chat API."""

import argparse
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ANSWER_PATTERN = re.compile(r"(?:^|\b)([A-Z0-9])(?:\b|$)")
RECIPIENT_HEADER = re.compile(r"^\s*to=user", re.IGNORECASE)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"{url} returned HTTP {error.code}: {body}") from error


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    labels = [str(label).upper() for label in row["choices"]["label"]]
    texts = row["choices"]["text"]
    choices = "\n".join(
        f"{label}. {text}" for label, text in zip(labels, texts, strict=True)
    )
    prompt = (
        "Answer this multiple-choice science question. Respond with only the "
        "letter of the correct answer.\n\n"
        f"Question: {row['question']}\n{choices}\n\nAnswer:"
    )
    return prompt, str(row["answerKey"]).upper()


def extract_answer(content: str, valid_labels: set[str]) -> str | None:
    normalized = RECIPIENT_HEADER.sub("", content, count=1).strip()
    for match in ANSWER_PATTERN.finditer(normalized.upper()):
        answer = match.group(1)
        if answer in valid_labels:
            return answer
    return None


def evaluate_row(
    index: int,
    row: dict[str, Any],
    base_url: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    prompt, expected = build_prompt(row)
    result = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 8,
        },
        timeout,
    )
    content = result["choices"][0]["message"]["content"] or ""
    labels = {str(label).upper() for label in row["choices"]["label"]}
    predicted = extract_answer(content, labels)
    return {
        "index": index,
        "id": row["id"],
        "expected": expected,
        "predicted": predicted,
        "content": content,
        "correct": predicted == expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31888")
    parser.add_argument("--model", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split=args.split,
    )
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    rows = [dataset[index] for index in range(total)]
    results: list[dict[str, Any] | None] = [None] * total

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                evaluate_row,
                index,
                row,
                args.base_url.rstrip("/"),
                args.model,
                args.timeout,
            ): index
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            results[index] = future.result()
            if completed % 50 == 0 or completed == total:
                correct = sum(
                    bool(result and result["correct"]) for result in results
                )
                print(f"{completed}/{total} complete, correct={correct}", flush=True)

    finalized = [result for result in results if result is not None]
    correct = sum(result["correct"] for result in finalized)
    invalid = sum(result["predicted"] is None for result in finalized)
    report = {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": args.split,
        "prompt_style": "zero-shot-chat-answer-letter-only",
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total,
        "results": finalized,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))


if __name__ == "__main__":
    main()
