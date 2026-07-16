#!/usr/bin/env python3
"""Deterministic correctness smoke test for a running Onyx SGLang server."""

import argparse
import json
import math
import re
import urllib.error
import urllib.request
from typing import Any


RAW_PARITY_INPUT_IDS = [200000, 954, 10810, 323, 4302, 373]
RAW_PARITY_FIRST_TOKEN = 328
CHAT_CASES = (
    ("What is the capital of France?", "paris"),
    ("中国的首都是哪里？", "北京"),
)
BOUNDARY_SEED = (
    "This is ordinary context used to exercise the sliding-window boundary. "
)
RECIPIENT_HEADER = re.compile(r"^\s*to=user", re.IGNORECASE)
ONYX_SERVER_PORT = 31888


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


def strip_recipient_header(text: str) -> str:
    return RECIPIENT_HEADER.sub("", text, count=1).lstrip()


def get_output_ids(result: dict[str, Any], tokenizer: Any) -> list[int]:
    output_ids = result.get("output_ids")
    if output_ids is not None:
        return list(output_ids)
    text = result.get("text")
    if not isinstance(text, str):
        raise AssertionError(f"Response has neither output_ids nor text: {result}")
    return list(tokenizer.encode(text, add_special_tokens=False))


def run_generate(
    base_url: str,
    input_ids: list[int],
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    return post_json(
        f"{base_url}/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout,
    )


def build_boundary_input(tokenizer: Any, target_len: int) -> list[int]:
    if target_len < 2:
        raise ValueError(f"Boundary input length must be at least 2, got {target_len}")
    seed_ids = list(tokenizer.encode(BOUNDARY_SEED, add_special_tokens=False))
    if not seed_ids:
        raise AssertionError("Boundary seed encoded to zero tokens")
    bos_token_id = tokenizer.bos_token_id
    prefix = [] if bos_token_id is None else [bos_token_id]
    repeat_count = math.ceil((target_len - len(prefix)) / len(seed_ids))
    return (prefix + seed_ids * repeat_count)[:target_len]


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",") if item]
    if not lengths:
        raise argparse.ArgumentTypeError("At least one boundary length is required")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default=f"http://localhost:{ONYX_SERVER_PORT}"
    )
    parser.add_argument("--model", default="default")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--chat-max-tokens", type=int, default=32)
    parser.add_argument(
        "--boundary-lengths",
        type=parse_lengths,
        default=parse_lengths("2047,2048,2049"),
    )
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    base_url = args.base_url.rstrip("/")
    summary: dict[str, Any] = {"raw": {}, "chat": [], "boundary": []}

    raw_result = run_generate(
        base_url,
        RAW_PARITY_INPUT_IDS,
        max_new_tokens=1,
        timeout=args.timeout,
    )
    raw_output_ids = get_output_ids(raw_result, tokenizer)
    if raw_output_ids != [RAW_PARITY_FIRST_TOKEN]:
        raise AssertionError(
            f"Raw parity expected [{RAW_PARITY_FIRST_TOKEN}], got {raw_output_ids}"
        )
    summary["raw"] = {
        "input_ids": RAW_PARITY_INPUT_IDS,
        "output_ids": raw_output_ids,
        "text": raw_result.get("text"),
    }

    for prompt, expected in CHAT_CASES:
        chat_result = post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": args.chat_max_tokens,
            },
            args.timeout,
        )
        content = chat_result["choices"][0]["message"]["content"]
        normalized = strip_recipient_header(content)
        if expected.casefold() not in normalized.casefold():
            raise AssertionError(
                f"Chat response for {prompt!r} does not contain {expected!r}: "
                f"{content!r}"
            )
        summary["chat"].append(
            {
                "prompt": prompt,
                "raw_content": content,
                "normalized_content": normalized,
            }
        )

    if not args.skip_boundary:
        for target_len in args.boundary_lengths:
            input_ids = build_boundary_input(tokenizer, target_len)
            boundary_result = run_generate(
                base_url,
                input_ids,
                max_new_tokens=1,
                timeout=args.timeout,
            )
            output_ids = get_output_ids(boundary_result, tokenizer)
            if len(output_ids) != 1:
                raise AssertionError(
                    f"Boundary length {target_len} returned {len(output_ids)} tokens"
                )
            if not 0 <= output_ids[0] < tokenizer.vocab_size:
                raise AssertionError(
                    f"Boundary length {target_len} returned invalid token {output_ids[0]}"
                )
            summary["boundary"].append(
                {
                    "input_len": len(input_ids),
                    "output_id": output_ids[0],
                    "text": boundary_result.get("text"),
                }
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
