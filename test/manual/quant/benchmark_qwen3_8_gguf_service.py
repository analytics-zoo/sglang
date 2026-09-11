#!/usr/bin/env python3
"""Small reproducible correctness/performance suite for Qwen3.8 GGUF E2E A/B."""

import argparse
import concurrent.futures
import json
import re
import statistics
import time
import urllib.request
from datetime import datetime, timezone


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _payload(model, prompt, max_tokens, stream=False):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _post(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
        status = response.status
    return status, time.perf_counter() - start, body


def _stream(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token = None
    content = []
    finish_reason = None
    with urllib.request.urlopen(request, timeout=300) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choice = event["choices"][0]
            token = choice.get("delta", {}).get("content") or ""
            if token and first_token is None:
                first_token = time.perf_counter()
            content.append(token)
            finish_reason = choice.get("finish_reason") or finish_reason
    end = time.perf_counter()
    return {
        "status": status,
        "elapsed_s": end - start,
        "ttft_s": None if first_token is None else first_token - start,
        "finish_reason": finish_reason,
        "content": "".join(content),
    }


def _record(kind, **values):
    print(json.dumps({"kind": kind, "utc": _utc(), **values}, ensure_ascii=False), flush=True)


def _completion(url, model, name, prompt, max_tokens):
    status, elapsed, body = _post(url, _payload(model, prompt, max_tokens))
    choice = body["choices"][0]
    usage = body.get("usage", {})
    content = choice["message"].get("content") or ""
    return {
        "name": name,
        "status": status,
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
    }


def _number_prefix_is_sequential(content, count=40):
    numbers = [int(value) for value in re.findall(r"\d+", content)]
    return numbers[:count] == list(range(1, count + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30001/v1/chat/completions")
    parser.add_argument("--model", default="/models/Qwen3.8-27B")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    correctness = (
        ("arith", "1+1 等于多少？只回答结果。", 32),
        ("multistep", "17*23-19 等于多少？只回答结果。", 32),
        ("json", '只输出严格 JSON：{"answer":7,"ok":true}', 64),
        ("code", "写一个简短的 Python 素数判断函数，只输出代码。", 128),
        ("fact", "法国的首都是什么？只回答城市名。", 32),
    )
    for name, prompt, max_tokens in correctness:
        _record("correctness", label=args.label, **_completion(
            args.url, args.model, name, prompt, max_tokens
        ))

    ttfts = []
    for run in range(1, 4):
        result = _stream(
            args.url,
            _payload(
                args.model,
                "为什么晴朗天空通常呈蓝色？请用一句话回答。",
                64,
                stream=True,
            ),
        )
        ttfts.append(result["ttft_s"])
        _record("ttft", label=args.label, run=run, **result)

    number_prompt = (
        "从1开始按顺序输出正整数，用英文逗号分隔，不要解释，"
        "不要停止，直到达到输出长度限制。"
    )
    decode_rates = []
    for run in range(1, 4):
        result = _completion(
            args.url, args.model, "decode256", number_prompt, 256
        )
        tokens = result["completion_tokens"] or 0
        rate = tokens / result["elapsed_s"]
        decode_rates.append(rate)
        _record(
            "decode256",
            label=args.label,
            run=run,
            tok_s=rate,
            prefix_sequential=_number_prefix_is_sequential(result["content"]),
            **{**result, "content": result["content"][:160]},
        )

    marker_rates = []
    marker_prompt = (
        ("alpha " * 2800)
        + "\nThe unique marker is BLUE-7391. Output only that marker."
    )
    for run in range(1, 4):
        result = _completion(
            args.url, args.model, "marker_prefill", marker_prompt, 32
        )
        prompt_tokens = result["prompt_tokens"] or 0
        rate = prompt_tokens / result["elapsed_s"]
        marker_rates.append(rate)
        _record(
            "marker_prefill",
            label=args.label,
            run=run,
            prompt_tok_s=rate,
            correct=result["content"].strip() == "BLUE-7391",
            **result,
        )

    for concurrency in (2, 4):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(concurrency) as executor:
            futures = [
                executor.submit(
                    _completion,
                    args.url,
                    args.model,
                    f"concurrency_{concurrency}_{index}",
                    number_prompt,
                    128,
                )
                for index in range(concurrency)
            ]
            results = [future.result() for future in futures]
        wall = time.perf_counter() - started
        total_tokens = sum(result["completion_tokens"] or 0 for result in results)
        _record(
            "concurrency",
            label=args.label,
            n=concurrency,
            wall_s=wall,
            total_completion_tokens=total_tokens,
            aggregate_tok_s=total_tokens / wall,
            statuses=[result["status"] for result in results],
            prefix_sequential=[
                _number_prefix_is_sequential(result["content"], 20)
                for result in results
            ],
            snippets=[result["content"][:100] for result in results],
        )

    long_prompt = (
        ("beta " * 18800)
        + "\nThe unique marker is ORANGE-86420. Output only that marker."
    )
    result = _completion(args.url, args.model, "long_context", long_prompt, 32)
    _record(
        "long_context",
        label=args.label,
        correct=result["content"].strip() == "ORANGE-86420",
        **result,
    )

    _record("summary", label=args.label, metric="ttft", values=ttfts,
            median=statistics.median(ttfts))
    _record("summary", label=args.label, metric="decode256", values=decode_rates,
            median=statistics.median(decode_rates))
    _record("summary", label=args.label, metric="marker_prefill", values=marker_rates,
            median=statistics.median(marker_rates))


if __name__ == "__main__":
    main()
