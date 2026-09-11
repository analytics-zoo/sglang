#!/usr/bin/env python3
"""Manual cached-vs-cold probe for generated-prefix Mamba/GDN state.

This script does not launch or manage a server.  It sends raw ``/generate``
requests to an already-running server and writes complete request/response
records to JSONL.  The generated-prefix case is deliberately different from
an ordinary prompt-cache hit: a first request generates through one or more
Mamba track boundaries, then a second request uses those exact generated token
IDs in its input prefix and leaves a short fresh suffix for prefill.

Example (the server in this experiment uses a 64-token track interval):

  python test/manual/quant/validate_qwen_gdn_generated_prefix.py \
    --tokenizer /models/Qwen3.8-27B --track-interval 64

The comparison is observational.  It records token and logprob differences,
but intentionally does not assert an arbitrary floating-point tolerance or
claim that cached and cold results must be bit-identical.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


DEFAULT_URL = "http://127.0.0.1:30001/generate"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonlLog:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", encoding="utf-8")

    def write(self, record):
        self.file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def error_dict(error):
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = error.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        return {"type": "http_error", "status": error.code, "message": str(error), "body": body}
    if isinstance(error, TimeoutError) or (
        isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)
    ):
        return {"type": "timeout", "message": str(error)}
    return {"type": type(error).__name__, "message": str(error)}


def request_json(url, payload, timeout, method="POST", require_json=True):
    data = None if method == "GET" else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
        result = {"status": status, "elapsed_s": time.perf_counter() - started}
        if not require_json:
            # /flush_cache intentionally returns human-readable plain text.
            result["raw_response"] = raw
            return result
        try:
            result["response"] = json.loads(raw)
        except json.JSONDecodeError as error:
            result["raw_response"] = raw
            result["error"] = {"type": "invalid_json", "message": str(error)}
        return result
    except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
        return {"elapsed_s": time.perf_counter() - started, "error": error_dict(error)}


def flush_cache(url, timeout):
    """Flush only when this manual validator is actually run."""
    if not url.rstrip("/").endswith("/generate"):
        raise ValueError("--url must name the raw /generate endpoint")
    result = request_json(
        url.rstrip("/").removesuffix("/generate") + "/flush_cache",
        {},
        timeout,
        require_json=False,
    )
    body = result.get("raw_response", "")
    result["flush_confirmed"] = result.get("status") == 200 and "Cache flushed" in body
    return result


def generate_payload(input_ids, max_new_tokens, return_logprob=True):
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": return_logprob,
        "top_logprobs_num": 5 if return_logprob else 0,
    }


def meta_info(result):
    response = result.get("response")
    return response.get("meta_info", {}) if isinstance(response, dict) else {}


def output_token_ids(result):
    """Use returned logprob token IDs; raw /generate need not expose output_ids."""
    response = result.get("response")
    if isinstance(response, dict) and isinstance(response.get("output_ids"), list):
        return [int(token_id) for token_id in response["output_ids"]]
    ids = []
    for item in meta_info(result).get("output_token_logprobs", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            ids.append(int(item[1]))
    return ids


def first_token_observation(result):
    meta = meta_info(result)
    token_logprobs = meta.get("output_token_logprobs") or []
    top_logprobs = meta.get("output_top_logprobs") or []
    return {
        "token_logprob": token_logprobs[0] if token_logprobs else None,
        "top_logprobs": top_logprobs[0] if top_logprobs else None,
    }


def cached_tokens(result):
    value = meta_info(result).get("cached_tokens")
    return int(value) if isinstance(value, int) else None


def run_generate(log, label, url, payload, timeout):
    result = request_json(url, payload, timeout)
    record = {
        "kind": "generate",
        "label": label,
        "utc": utc_now(),
        "request": payload,
        "result": result,
        "cached_tokens": cached_tokens(result),
        "output_token_ids": output_token_ids(result),
        "first_token": first_token_observation(result),
    }
    log.write(record)
    return record


def run_flush(log, url, timeout, label):
    result = flush_cache(url, timeout)
    record = {"kind": "flush_cache", "label": label, "utc": utc_now(), "result": result}
    log.write(record)
    return record


def is_success(record):
    return record["result"].get("status") == 200 and "error" not in record["result"]


def compare_cached_and_cold(cached, cold, initial_prompt_length):
    cached_ids = cached["output_token_ids"]
    cold_ids = cold["output_token_ids"]
    cached_first = cached["first_token"]
    cold_first = cold["first_token"]
    cached_logprob = cached_first["token_logprob"]
    cold_logprob = cold_first["token_logprob"]
    same_first_token_id = bool(cached_ids and cold_ids and cached_ids[0] == cold_ids[0])
    logprob_delta = None
    if (
        same_first_token_id
        and
        isinstance(cached_logprob, (list, tuple))
        and isinstance(cold_logprob, (list, tuple))
        and cached_logprob
        and cold_logprob
        and isinstance(cached_logprob[0], (float, int))
        and isinstance(cold_logprob[0], (float, int))
    ):
        logprob_delta = float(cached_logprob[0]) - float(cold_logprob[0])
    return {
        "cached_tokens": cached["cached_tokens"],
        "cached_tokens_exceed_initial_prompt": (
            cached["cached_tokens"] is not None
            and cached["cached_tokens"] > initial_prompt_length
        ),
        "cached_first_token_id": cached_ids[0] if cached_ids else None,
        "cold_first_token_id": cold_ids[0] if cold_ids else None,
        "first_token_id_equal": same_first_token_id,
        "cached_first_token_logprob": cached_logprob,
        "cold_first_token_logprob": cold_logprob,
        "first_token_logprob_delta_cached_minus_cold": logprob_delta,
        "first_token_logprob_delta_note": (
            "computed for the same chosen token"
            if same_first_token_id
            else "not computed because cached and cold chose different first token IDs"
        ),
        "cached_first_top_logprobs": cached_first["top_logprobs"],
        "cold_first_top_logprobs": cold_first["top_logprobs"],
        "first_top_logprobs_exactly_equal": cached_first["top_logprobs"] == cold_first["top_logprobs"],
        "generated_token_ids_equal": cached_ids == cold_ids,
        "cached_completion_token_count": len(cached_ids),
        "cold_completion_token_count": len(cold_ids),
        "note": "Observations only: no floating-point tolerance or bit-exact correctness threshold is asserted.",
    }


def load_tokenizer(path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("This manual probe requires transformers for exact HF tokenization.") from error
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def build_aligned_prompt(tokenizer, track_interval):
    seed = tokenizer.encode(
        "A lighthouse keeper writes a precise weather log before dawn.",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(" the", add_special_tokens=False)
    if not seed or len(filler) != 1:
        raise RuntimeError("Expected a nonempty seed and a one-token HF filler for deterministic raw input_ids.")
    aligned_length = ((len(seed) + track_interval - 1) // track_interval) * track_interval
    return seed + [filler[0]] * (aligned_length - len(seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Raw SGLang /generate endpoint")
    parser.add_argument("--tokenizer", default="/models/Qwen3.8-27B", help="HF tokenizer path")
    parser.add_argument("--track-interval", type=int, default=64, help="Server --mamba-track-interval")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--fresh-suffix-tokens", type=int, default=16)
    parser.add_argument("--compare-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default=None, help="JSONL path; defaults to /tmp with UTC timestamp")
    args = parser.parse_args()

    if args.track_interval <= 0 or args.max_new_tokens < args.track_interval:
        parser.error("--track-interval must be positive and --max-new-tokens must reach at least one boundary")
    if args.max_new_tokens % args.track_interval:
        parser.error("--max-new-tokens must be a multiple of --track-interval for a reproducible boundary-crossing probe")
    if args.fresh_suffix_tokens <= 0 or args.compare_new_tokens <= 0:
        parser.error("suffix and comparison token counts must be positive")

    tokenizer = load_tokenizer(args.tokenizer)
    initial_prompt = build_aligned_prompt(tokenizer, args.track_interval)
    fresh_ids = tokenizer.encode(" continuing carefully", add_special_tokens=False)
    if not fresh_ids:
        raise RuntimeError("HF tokenizer produced no continuation tokens")
    fresh_suffix = (fresh_ids * ((args.fresh_suffix_tokens + len(fresh_ids) - 1) // len(fresh_ids)))[: args.fresh_suffix_tokens]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or "/tmp/qwen_gdn_generated_prefix_%s.jsonl" % timestamp
    log = JsonlLog(output)
    records = []
    try:
        records.append(run_flush(log, args.url, args.timeout, "before_generated_producer"))
        producer = run_generate(
            log,
            "generated_prefix_producer",
            args.url,
            generate_payload(initial_prompt, args.max_new_tokens),
            args.timeout,
        )
        records.append(producer)
        generated_ids = producer["output_token_ids"]

        generated_case = None
        cold = None
        if is_success(producer) and len(generated_ids) == args.max_new_tokens:
            continuation_prefix = initial_prompt + generated_ids + fresh_suffix
            cached = run_generate(
                log,
                "generated_prefix_cached",
                args.url,
                generate_payload(continuation_prefix, args.compare_new_tokens),
                args.timeout,
            )
            records.append(cached)
            records.append(run_flush(log, args.url, args.timeout, "before_generated_prefix_cold"))
            cold = run_generate(
                log,
                "generated_prefix_cold",
                args.url,
                generate_payload(continuation_prefix, args.compare_new_tokens),
                args.timeout,
            )
            records.append(cold)
            generated_case = compare_cached_and_cold(cached, cold, len(initial_prompt))
            generated_case.update(
                {
                    "input_prefix_length": len(continuation_prefix),
                    "initial_prompt_length": len(initial_prompt),
                    "generated_prefix_length": len(initial_prompt) + len(generated_ids),
                    "generated_token_count": len(generated_ids),
                    "fresh_suffix_token_count": len(fresh_suffix),
                }
            )
        else:
            generated_case = {
                "skipped": True,
                "reason": "producer did not return the requested exact generated token sequence",
                "requested_generated_tokens": args.max_new_tokens,
                "received_generated_tokens": len(generated_ids),
            }

        # Control: cache only an aligned user-prefill prefix, then compare its
        # cached continuation with the same continuation after a flush.
        records.append(run_flush(log, args.url, args.timeout, "before_prefill_control"))
        control_producer = run_generate(
            log,
            "prefill_control_producer",
            args.url,
            generate_payload(initial_prompt, 1, return_logprob=False),
            args.timeout,
        )
        records.append(control_producer)
        control_prefix = initial_prompt + fresh_suffix
        control_cached = run_generate(
            log,
            "prefill_control_cached",
            args.url,
            generate_payload(control_prefix, args.compare_new_tokens),
            args.timeout,
        )
        records.append(control_cached)
        records.append(run_flush(log, args.url, args.timeout, "before_prefill_control_cold"))
        control_cold = run_generate(
            log,
            "prefill_control_cold",
            args.url,
            generate_payload(control_prefix, args.compare_new_tokens),
            args.timeout,
        )
        records.append(control_cold)

        generated_cached_hit = (
            generated_case.get("cached_tokens_exceed_initial_prompt")
            if isinstance(generated_case, dict)
            else False
        )
        generated_cold_uncached = cold is not None and cold["cached_tokens"] == 0
        control_cached_hit = control_cached["cached_tokens"] is not None and control_cached["cached_tokens"] > 0
        control_cold_uncached = control_cold["cached_tokens"] == 0
        flush_confirmed = all(
            record["result"].get("flush_confirmed") is True
            for record in records
            if record["kind"] == "flush_cache"
        )
        checks = {
            "all_flushes_confirmed": flush_confirmed,
            "generated_prefix_cached_tokens_exceed_initial_prompt": generated_cached_hit,
            "generated_prefix_cold_cached_tokens_zero": generated_cold_uncached,
            "prefill_control_cached_tokens_positive": control_cached_hit,
            "prefill_control_cold_cached_tokens_zero": control_cold_uncached,
        }
        summary = {
            "kind": "summary",
            "utc": utc_now(),
            "output": output,
            "url": args.url,
            "tokenizer": args.tokenizer,
            "track_interval": args.track_interval,
            "initial_prompt_length": len(initial_prompt),
            "generated_prefix_case": generated_case,
            "prefill_cache_control": compare_cached_and_cold(
                control_cached, control_cold, len(initial_prompt)
            ),
            "checks": checks,
            "inconclusive": not all(checks.values()),
            "transport_failures": sum(not is_success(record) for record in records),
        }
        log.write(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if summary["transport_failures"] == 0 and not summary["inconclusive"] else 2
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
