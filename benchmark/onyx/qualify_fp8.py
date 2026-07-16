#!/usr/bin/env python3
"""Capture and compare Onyx FP16/FP8 model-level correctness suites."""

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from validate_onyx import (
    BOUNDARY_SEED,
    ONYX_SERVER_PORT,
    RAW_PARITY_FIRST_TOKEN,
    RAW_PARITY_INPUT_IDS,
    build_boundary_input,
    get_output_ids,
    parse_lengths,
    strip_recipient_header,
)


RAW_TEXTS = (
    "The quick brown fox jumps over the lazy dog.",
    "请用一句话解释为什么天空看起来是蓝色的。",
    "def fibonacci(n):",
    "Symbols: α β γ, JSON {\"value\": 42}, hexadecimal 0xBEEF.",
)
TASK_CASES = (
    ("What is the capital of France?", "paris"),
    ("中国的首都是哪里？", "北京"),
    ("What is 17 multiplied by 23? Answer with the number.", "391"),
    ("Compute 144 divided by 12. Answer with the number.", "12"),
    ("If Alice has 9 apples and gives away 4, how many remain?", "5"),
    ("Name the largest planet in our solar system.", "jupiter"),
    ("What chemical symbol represents water?", "h2o"),
    ("将数字二十三写成阿拉伯数字。", "23"),
    ("Which is larger, 0.75 or 0.7?", "0.75"),
    ("Continue the sequence 2, 4, 8, 16 with one number.", "32"),
)
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


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


def assert_finite(value: Any, path: str = "response") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"{path} contains non-finite value {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{path}[{index}]")


def generate_with_logprobs(
    base_url: str,
    input_ids: list[int],
    timeout: float,
) -> dict[str, Any]:
    result = post_json(
        f"{base_url}/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 5,
            "return_text_in_logprobs": True,
        },
        timeout,
    )
    assert_finite(result)
    meta = result["meta_info"]
    output_top = meta["output_top_logprobs"]
    if len(output_top) != 1 or len(output_top[0]) != 5:
        raise AssertionError(f"Expected one top-5 output row, got {output_top}")
    return result


def chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    **extra: Any,
) -> dict[str, Any]:
    result = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            **extra,
        },
        timeout,
    )
    assert_finite(result)
    return result


def capture(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    base_url = args.base_url.rstrip("/")
    report: dict[str, Any] = {
        "label": args.label,
        "radix_cache": args.radix_cache,
        "raw": [],
        "tasks": [],
        "boundaries": [],
        "multi_turn": {},
        "prefix_repeat": {},
        "tool_call": {},
    }

    raw_cases = [("fixed", RAW_PARITY_INPUT_IDS)]
    raw_cases.extend(
        (f"text_{index}", tokenizer.encode(text, add_special_tokens=True))
        for index, text in enumerate(RAW_TEXTS)
    )
    for name, input_ids in raw_cases:
        result = generate_with_logprobs(base_url, list(input_ids), args.timeout)
        output_ids = get_output_ids(result, tokenizer)
        if name == "fixed" and output_ids != [RAW_PARITY_FIRST_TOKEN]:
            raise AssertionError(
                f"Fixed raw prompt expected token {RAW_PARITY_FIRST_TOKEN}, "
                f"got {output_ids}"
            )
        report["raw"].append(
            {
                "name": name,
                "input_ids": input_ids,
                "output_ids": output_ids,
                "text": result.get("text"),
                "output_token_logprobs": result["meta_info"][
                    "output_token_logprobs"
                ],
                "output_top_logprobs": result["meta_info"][
                    "output_top_logprobs"
                ],
                "prompt_token_count": len(
                    result["meta_info"]["input_token_logprobs"]
                ),
            }
        )

    for prompt, expected in TASK_CASES:
        result = chat(
            base_url,
            args.model,
            [{"role": "user", "content": prompt}],
            args.chat_max_tokens,
            args.timeout,
        )
        content = result["choices"][0]["message"]["content"] or ""
        normalized = strip_recipient_header(content)
        passed = expected.casefold() in normalized.casefold()
        report["tasks"].append(
            {
                "prompt": prompt,
                "expected": expected,
                "content": content,
                "normalized_content": normalized,
                "passed": passed,
            }
        )
    report["task_score"] = sum(row["passed"] for row in report["tasks"]) / len(
        report["tasks"]
    )

    for target_len in args.boundary_lengths:
        input_len = (
            target_len - 1 if target_len == args.max_context_length else target_len
        )
        input_ids = build_boundary_input(tokenizer, input_len)
        result = generate_with_logprobs(base_url, input_ids, args.timeout)
        output_ids = get_output_ids(result, tokenizer)
        if len(output_ids) != 1 or not 0 <= output_ids[0] < tokenizer.vocab_size:
            raise AssertionError(
                f"Length {target_len} returned invalid output IDs {output_ids}"
            )
        report["boundaries"].append(
            {
                "target_context_len": target_len,
                "input_len": len(input_ids),
                "total_len": len(input_ids) + len(output_ids),
                "output_ids": output_ids,
                "text": result.get("text"),
                "output_token_logprobs": result["meta_info"][
                    "output_token_logprobs"
                ],
                "output_top_logprobs": result["meta_info"][
                    "output_top_logprobs"
                ],
            }
        )

    multi_turn = chat(
        base_url,
        args.model,
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital is Paris."},
            {"role": "user", "content": "Which country is that city in?"},
        ],
        32,
        args.timeout,
    )
    multi_turn_content = multi_turn["choices"][0]["message"]["content"] or ""
    report["multi_turn"] = {
        "content": multi_turn_content,
        "passed": "france" in multi_turn_content.casefold(),
    }

    prefix_ids = build_boundary_input(tokenizer, 4096)
    prefix_outputs = []
    for _ in range(2):
        result = post_json(
            f"{base_url}/generate",
            {
                "input_ids": prefix_ids,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 4},
            },
            args.timeout,
        )
        assert_finite(result)
        prefix_outputs.append(get_output_ids(result, tokenizer))
    report["prefix_repeat"] = {
        "input_len": len(prefix_ids),
        "outputs": prefix_outputs,
        "repeatable": prefix_outputs[0] == prefix_outputs[1],
    }

    tool_result = chat(
        base_url,
        args.model,
        [{"role": "user", "content": "What is the weather in Paris?"}],
        64,
        args.timeout,
        tools=[TOOL],
        tool_choice="auto",
    )
    message = tool_result["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    tool_passed = False
    if tool_calls:
        function = tool_calls[0].get("function", {})
        arguments = function.get("arguments", "")
        tool_passed = (
            function.get("name") == "get_weather"
            and "paris" in str(arguments).casefold()
        )
    report["tool_call"] = {
        "content": message.get("content"),
        "tool_calls": tool_calls,
        "passed": tool_passed,
    }

    if not report["multi_turn"]["passed"]:
        raise AssertionError(f"Multi-turn check failed: {report['multi_turn']}")
    if not report["prefix_repeat"]["repeatable"]:
        raise AssertionError(f"Prefix repeat failed: {report['prefix_repeat']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def top_ids(row: dict[str, Any]) -> list[int]:
    return [int(item[1]) for item in row["output_top_logprobs"][0]]


def compare(args: argparse.Namespace) -> None:
    reference = json.loads(Path(args.reference).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    result: dict[str, Any] = {
        "reference": args.reference,
        "candidate": args.candidate,
        "raw": [],
        "boundaries": [],
    }

    if reference["radix_cache"] != candidate["radix_cache"]:
        raise AssertionError("Cannot compare captures with different radix modes")
    for reference_row, candidate_row in zip(
        reference["raw"], candidate["raw"], strict=True
    ):
        if reference_row["name"] != candidate_row["name"]:
            raise AssertionError("Raw capture order differs")
        reference_top = top_ids(reference_row)
        candidate_top = top_ids(candidate_row)
        row = {
            "name": reference_row["name"],
            "reference_output_ids": reference_row["output_ids"],
            "candidate_output_ids": candidate_row["output_ids"],
            "top_token_match": reference_top[0] == candidate_top[0],
            "top5_overlap": len(set(reference_top) & set(candidate_top)),
        }
        result["raw"].append(row)

    reference_boundaries = {
        row["input_len"]: row["output_ids"] for row in reference["boundaries"]
    }
    candidate_boundaries = {
        row["input_len"]: row["output_ids"] for row in candidate["boundaries"]
    }
    for length in reference_boundaries:
        result["boundaries"].append(
            {
                "input_len": length,
                "reference_output_ids": reference_boundaries[length],
                "candidate_output_ids": candidate_boundaries.get(length),
                "match": reference_boundaries[length]
                == candidate_boundaries.get(length),
            }
        )

    result.update(
        {
            "reference_task_score": reference["task_score"],
            "candidate_task_score": candidate["task_score"],
            "task_score_delta": candidate["task_score"] - reference["task_score"],
            "multi_turn_passed": candidate["multi_turn"]["passed"],
            "prefix_repeatable": candidate["prefix_repeat"]["repeatable"],
            "tool_call_passed": candidate["tool_call"]["passed"],
        }
    )
    result["passed"] = (
        all(row["top_token_match"] for row in result["raw"])
        and all(row["match"] for row in result["boundaries"])
        and result["task_score_delta"] >= -0.01
        and result["multi_turn_passed"]
        and result["prefix_repeatable"]
        and result["tool_call_passed"]
    )
    if args.reference_gsm8k is not None and args.candidate_gsm8k is not None:
        result["reference_gsm8k"] = args.reference_gsm8k
        result["candidate_gsm8k"] = args.candidate_gsm8k
        result["gsm8k_delta"] = args.candidate_gsm8k - args.reference_gsm8k
        result["passed"] = result["passed"] and result["gsm8k_delta"] >= -0.0100001

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument(
        "--base-url", default=f"http://localhost:{ONYX_SERVER_PORT}"
    )
    capture_parser.add_argument("--model", default="default")
    capture_parser.add_argument("--tokenizer-path", required=True)
    capture_parser.add_argument("--label", required=True)
    capture_parser.add_argument("--radix-cache", choices=("on", "off"), required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--chat-max-tokens", type=int, default=64)
    capture_parser.add_argument(
        "--boundary-lengths",
        type=parse_lengths,
        default=parse_lengths("2047,2048,2049,4096,8192"),
    )
    capture_parser.add_argument("--max-context-length", type=int, default=16384)
    capture_parser.add_argument("--timeout", type=float, default=600.0)
    capture_parser.set_defaults(function=capture)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--reference-gsm8k", type=float)
    compare_parser.add_argument("--candidate-gsm8k", type=float)
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(function=compare)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
