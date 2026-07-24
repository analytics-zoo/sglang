#!/usr/bin/env python3
"""Run the 20k.json long-history agent continuation through SGLang Chat API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def normalize_onyx_tool_history(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Split historical parallel tool calls into sequential Onyx tool turns."""
    normalized: list[dict[str, Any]] = []
    split_turns = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or len(tool_calls) <= 1:
            normalized.append(message)
            index += 1
            continue

        results: list[dict[str, Any]] = []
        next_index = index + 1
        while next_index < len(messages) and messages[next_index].get("role") == "tool":
            results.append(messages[next_index])
            next_index += 1
        results_by_id = {result.get("tool_call_id"): result for result in results}
        if None in results_by_id or len(results_by_id) != len(results):
            raise ValueError(
                f"parallel tool turn at message {index} has ambiguous results"
            )

        used_ids: set[str] = set()
        for call_index, tool_call in enumerate(tool_calls):
            call_id = tool_call.get("id")
            if not call_id or call_id not in results_by_id:
                raise ValueError(
                    f"parallel tool call at message {index} has no matching result"
                )
            assistant_turn = dict(message)
            assistant_turn["tool_calls"] = [tool_call]
            if call_index > 0 and "reasoning_content" in assistant_turn:
                assistant_turn["reasoning_content"] = ""
            normalized.extend((assistant_turn, results_by_id[call_id]))
            used_ids.add(call_id)
        if used_ids != set(results_by_id):
            raise ValueError(f"parallel tool turn at message {index} has extra results")
        split_turns += 1
        index = next_index
    return normalized, split_turns


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:4000]}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"expected object response, got {type(result).__name__}")
    return result


def parse_arguments(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    arguments = (tool_call.get("function") or {}).get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score_response(
    message: dict[str, Any], valid_tool_names: set[str]
) -> tuple[dict[str, bool], dict[str, Any] | None]:
    tool_calls = message.get("tool_calls") or []
    tool_call = tool_calls[0] if len(tool_calls) == 1 else None
    tool_name = (tool_call.get("function") or {}).get("name") if tool_call else None
    arguments = parse_arguments(tool_call) if tool_call else None
    command = arguments.get("command", "") if arguments else ""
    command_lower = command.casefold() if isinstance(command, str) else ""
    checks = {
        "exactly_one_tool_call": len(tool_calls) == 1,
        "tool_name_is_defined": bool(tool_name and tool_name in valid_tool_names),
        "arguments_are_json_object": arguments is not None,
        "continues_with_shell_command": tool_name == "run_shell_command",
        "targets_existing_project": "demo-game-snake-ovms" in command_lower,
        "installs_npm_dependency": "npm install" in command_lower,
        "installs_typescript": "typescript" in command_lower,
        "runs_in_foreground": bool(
            arguments is not None and arguments.get("is_background") is False
        ),
    }
    return checks, arguments


def write_report(report: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/llm/workspace/20k.json"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("/llm/workspace/model/onyx-hf")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:31888")
    parser.add_argument("--model", default="default")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    source_messages = source.get("messages")
    tools = source.get("tools")
    if not isinstance(source_messages, list) or not source_messages:
        raise ValueError("input must contain a non-empty messages array")
    if not isinstance(tools, list) or not tools:
        raise ValueError("input must contain a non-empty tools array")
    messages, split_turns = normalize_onyx_tool_history(source_messages)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    declared_limit = int(tokenizer.model_max_length)
    tokenizer.model_max_length = 10**9
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    valid_tool_names = {
        function["name"]
        for tool in tools
        if isinstance((function := tool.get("function")), dict)
        and isinstance(function.get("name"), str)
    }

    report: dict[str, Any] = {
        "test": "sglang_onyx_20k_agent_continuation",
        "input": str(args.input),
        "base_url": args.base_url,
        "source_message_count": len(source_messages),
        "normalized_message_count": len(messages),
        "split_parallel_tool_turns": split_turns,
        "tool_count": len(tools),
        "prompt_tokens": len(prompt_ids),
        "requested_total_tokens": len(prompt_ids) + args.max_tokens,
        "tokenizer_declared_model_max_length": declared_limit,
        "dry_run": args.dry_run,
        "passed": False,
    }
    if args.dry_run:
        report.update({"phase": "dry_run", "passed": True})
        write_report(report, args.output)
        return 0

    request_payload = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    start = time.perf_counter()
    try:
        result = post_json(
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            request_payload,
            args.timeout,
        )
        choice = result["choices"][0]
        message = choice["message"]
        checks, arguments = score_response(message, valid_tool_names)
        report.update(
            {
                "phase": "complete",
                "elapsed_seconds": time.perf_counter() - start,
                "finish_reason": choice.get("finish_reason"),
                "assistant_message": message,
                "parsed_arguments": arguments,
                "checks": checks,
                "passed": all(checks.values()),
                "usage": result.get("usage"),
            }
        )
    except Exception as error:
        report.update(
            {
                "phase": "error",
                "elapsed_seconds": time.perf_counter() - start,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    write_report(report, args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
