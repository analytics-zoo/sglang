#!/usr/bin/env python3
"""Long-running, manual transport-stability check for a Qwen3.8 GGUF server.

The client never manages the server.  It writes one JSON object per line, including
the complete request and response, so a failed run can be replayed from its log.
For example, a short smoke run is:

  python stress_qwen3_8_gguf_service.py --duration-seconds 60 --model /models/Qwen3.8-27B
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


DEFAULT_URL = "http://127.0.0.1:30001/v1/chat/completions"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def percentile(values, fraction):
    """Nearest-rank percentile without a third-party dependency."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))
    return ordered[index]


class JsonlLog:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, record):
        with self.lock:
            self.file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            self.file.flush()

    def close(self):
        self.file.close()


def request_payload(model, prompt, max_tokens, stream=False, ignore_eos=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        # SGLang accepts this OpenAI-compatible sampling parameter.  It makes a
        # decode workload much more likely to reach its configured token budget.
        payload["ignore_eos"] = True
    return payload


def error_dict(error):
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = error.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        return {"type": "http_error", "status": error.code, "message": str(error), "body": body}
    if isinstance(error, TimeoutError) or isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError):
        return {"type": "timeout", "message": str(error)}
    return {"type": type(error).__name__, "message": str(error)}


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            status = response.status
        elapsed = time.perf_counter() - started
        try:
            return {"status": status, "elapsed_s": elapsed, "response": json.loads(raw_body)}
        except json.JSONDecodeError as error:
            return {"status": status, "elapsed_s": elapsed, "raw_response": raw_body,
                    "error": {"type": "invalid_json", "message": str(error)}}
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        return {"elapsed_s": time.perf_counter() - started, "error": error_dict(error)}


def get_health(url, timeout):
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {"status": response.status, "elapsed_s": time.perf_counter() - started,
                    "body": body}
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        return {"elapsed_s": time.perf_counter() - started, "error": error_dict(error)}


def content_and_usage(result):
    response = result.get("response")
    if not isinstance(response, dict):
        return "", {}, None
    choice = {}
    try:
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content") or ""
    except (IndexError, KeyError, TypeError):
        content = ""
    return content, response.get("usage") or {}, choice.get("finish_reason")


def marker_prompt(word_count, marker, request_id):
    # Repeated ordinary words make the prefill length predictable without relying
    # on a tokenizer that might differ from the serving tokenizer.
    filler = "stability " * word_count
    return (
        # A unique prefix prevents a hit on an identical full prompt in a radix
        # cache.  The logged prefill mode makes this assumption explicit.
        f"Unique request nonce: {request_id}.\n{filler}\n"
        f'Return only strict JSON: {{"marker":"{marker}","ok":true}}\n'
        "Do not explain, quote, format, or add fields."
    )


def decode_prompt(word_count, request_id):
    # This is deliberately a quality-neutral request.  Nonempty completion is a
    # liveness check; marker cases above provide the strict content assertion.
    filler = "stability " * word_count
    return (
        f"Unique request nonce: {request_id}.\n{filler}\n"
        "Write a continuous, coherent travel diary about a lighthouse in a storm. "
        "Keep adding concrete details until the output limit; do not conclude early."
    )


def json_marker_check(content, marker):
    try:
        return json.loads(content.strip()) == {"marker": marker, "ok": True}
    except (TypeError, json.JSONDecodeError):
        return False


def run_case(url, model, phase, workload, request_id, concurrency, timeout, ignore_eos):
    words, max_tokens = phase
    marker = "STABILITY-%d-%s" % (request_id, "A7K9")
    if workload == "marker_json":
        payload = request_payload(model, marker_prompt(words, marker, request_id), max_tokens)
    else:
        payload = request_payload(
            model, decode_prompt(words, request_id), max_tokens, ignore_eos=ignore_eos
        )
    result = post_json(url, payload, timeout)
    content, usage, finish_reason = content_and_usage(result)
    check = (
        {"kind": "parsed_json_marker", "expected": {"marker": marker, "ok": True},
         "passed": json_marker_check(content, marker)}
        if workload == "marker_json" else
        {"kind": "nonempty_decode", "passed": bool(content.strip()),
         "note": "Liveness only; this client makes no semantic quality claim."}
    )
    result.update({
        "kind": "request",
        "utc": utc_now(),
        "request_id": request_id,
        "phase": "prefill_%dk_decode_%d" % (words // 1000, max_tokens),
        "workload": workload,
        "concurrency": concurrency,
        "prefill_mode": "unique_prefix_per_request",
        "decode_target_max_tokens": max_tokens,
        "finish_reason": finish_reason,
        "request": payload,
        "content_check": check,
        "content": content,
        "usage": usage,
    })
    return result


def cancel_stream(url, model, timeout):
    payload = request_payload(
        model, marker_prompt(4000, "CANCEL-STREAM-OK", "cancel-stream"), 256, stream=True
    )
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    event_count, partial = 0, []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event_count += 1
                try:
                    event = json.loads(line[6:])
                    partial.append(event.get("choices", [{}])[0].get("delta", {}).get("content") or "")
                except (json.JSONDecodeError, IndexError, AttributeError):
                    pass
                # Leaving this context closes the response and deliberately
                # cancels the request from the client's side.
                break
        return {"kind": "cancel_stream", "utc": utc_now(), "request": payload,
                "status": status, "elapsed_s": time.perf_counter() - started,
                "events_before_cancel": event_count, "partial_content": "".join(partial),
                "canceled_by_client": event_count > 0}
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        return {"kind": "cancel_stream", "utc": utc_now(), "request": payload,
                "elapsed_s": time.perf_counter() - started, "error": error_dict(error),
                "canceled_by_client": False}


def timeout_probe(url, model, timeout):
    payload = request_payload(model, marker_prompt(4000, "TIMEOUT-PROBE-OK", "timeout-probe"), 256)
    result = post_json(url, payload, timeout)
    content, _, _ = content_and_usage(result)
    result.update({"kind": "timeout_probe", "utc": utc_now(), "request": payload,
                   "content": content, "timeout_seconds": timeout,
                   # Completing within the guard is valid; timing out proves the
                   # client timeout path.  Other transport errors are reported.
                   "outcome": "timed_out" if result.get("error", {}).get("type") == "timeout"
                   else "completed" if result.get("status") == 200 else "error"})
    return result


def recovery_check(url, health_url, model, timeout):
    health = get_health(health_url, timeout)
    marker = "RECOVERY-MARKER-2049"
    payload = request_payload(model, "Return exactly %s and nothing else." % marker, 32)
    completion = post_json(url, payload, timeout)
    content, _, _ = content_and_usage(completion)
    return {
        "kind": "recovery", "utc": utc_now(), "health": health, "request": payload,
        "completion": completion, "content": content,
        "health_passed": health.get("status") == 200,
        "marker_passed": content.strip() == marker,
    }


def summary_view(record):
    """Keep summary bookkeeping bounded while JSONL retains full replay data."""
    kind = record.get("kind")
    if kind == "request":
        return {key: record.get(key) for key in (
            "kind", "phase", "workload", "concurrency", "status", "elapsed_s",
            "content_check", "usage", "finish_reason", "error",
        )}
    if kind == "recovery":
        return {"kind": kind, "health_passed": record["health_passed"],
                "marker_passed": record["marker_passed"],
                "health_error": record["health"].get("error"),
                "completion_error": record["completion"].get("error")}
    if kind in ("timeout_probe", "cancel_stream"):
        return {key: record.get(key) for key in (
            "kind", "outcome", "canceled_by_client", "error", "status",
        )}
    return {"kind": kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--health-url", help="Defaults to the server's /health endpoint.")
    parser.add_argument("--model", default="/models/Qwen3.8-27B")
    parser.add_argument("--duration-seconds", type=float, default=1800)
    parser.add_argument("--request-timeout-seconds", type=float, default=300)
    parser.add_argument("--timeout-probe-seconds", type=float, default=1)
    parser.add_argument("--skip-probes", action="store_true",
                        help="Skip timeout/cancel/initial-recovery probes and run sustained requests only.")
    parser.add_argument("--stop-on-transport-error", action="store_true",
                        help="Stop after the current workload batch if a request has a transport/protocol error.")
    parser.add_argument("--no-ignore-eos", action="store_true",
                        help="Allow normal EOS during decode workloads.")
    parser.add_argument("--output", help="JSONL output path (default includes UTC timestamp).")
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.request_timeout_seconds <= 0 or args.timeout_probe_seconds <= 0:
        parser.error("duration and timeout values must be positive")

    health_url = args.health_url or args.url.split("/v1/", 1)[0].rstrip("/") + "/health"
    output = args.output or "stress_qwen3_8_gguf_%s.jsonl" % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = JsonlLog(output)
    summary_records = []
    started = time.monotonic()
    timeout_probe_record = None
    cancel_stream_record = None
    stopped_on_transport_error = False
    try:
        log.write({"kind": "run_start", "utc": utc_now(), "url": args.url,
                   "health_url": health_url, "model": args.model,
                   "duration_seconds": args.duration_seconds,
                   "request_timeout_seconds": args.request_timeout_seconds,
                   "skip_probes": args.skip_probes,
                   "stop_on_transport_error": args.stop_on_transport_error})
        # Exercise timeout handling and an abandoned SSE connection before the
        # sustained workload, then require a normal health/completion recovery.
        if not args.skip_probes:
            timeout_probe_record = timeout_probe(
                args.url, args.model, args.timeout_probe_seconds
            )
            cancel_stream_record = cancel_stream(
                args.url, args.model, args.request_timeout_seconds
            )
            initial_recovery = recovery_check(
                args.url, health_url, args.model, args.request_timeout_seconds
            )
            for record in (timeout_probe_record, cancel_stream_record, initial_recovery):
                log.write(record)
                summary_records.append(summary_view(record))

        deadline = time.monotonic() + args.duration_seconds
        phases = ((1000, 128), (1000, 256), (4000, 128), (4000, 256))
        concurrencies = (1, 2, 4)
        request_id = 0
        batch = 0
        while time.monotonic() < deadline:
            concurrency = concurrencies[batch % len(concurrencies)]
            phase = phases[batch % len(phases)]
            batch_transport_error = False
            # Keep each individual workload at the advertised concurrency.  The
            # marker group checks deterministic content, then the decode group
            # exercises actual 128/256-token generation without a quality claim.
            for workload in ("marker_json", "decode"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for _ in range(concurrency):
                        request_id += 1
                        futures.append(executor.submit(
                            run_case, args.url, args.model, phase, workload, request_id,
                            concurrency, args.request_timeout_seconds, not args.no_ignore_eos,
                        ))
                    for future in futures:
                        record = future.result()
                        log.write(record)
                        summary_records.append(summary_view(record))
                        batch_transport_error |= bool(record.get("error"))
            batch += 1
            # Do not interrupt a concurrent group mid-flight: its records are
            # valuable for comparing which rows failed.  Avoid beginning the
            # next batch, whose request timeout could otherwise add minutes.
            if args.stop_on_transport_error and batch_transport_error:
                stopped_on_transport_error = True
                break

        # A dead service must not turn a transport-stop into two more 300-second
        # waits.  The final check is diagnostic and always recorded.
        final_recovery_timeout = min(args.request_timeout_seconds, 10.0)
        final_recovery = recovery_check(args.url, health_url, args.model, final_recovery_timeout)
        log.write(final_recovery)
        summary_records.append(summary_view(final_recovery))
        requests = [record for record in summary_records if record.get("kind") == "request"]
        phase_summary = {}
        for phase in sorted({(record["phase"], record["workload"]) for record in requests}):
            records = [record for record in requests if (record["phase"], record["workload"]) == phase]
            elapsed = [record["elapsed_s"] for record in records if "elapsed_s" in record]
            completion_tokens = [
                record.get("usage", {}).get("completion_tokens") for record in records
                if isinstance(record.get("usage", {}).get("completion_tokens"), int)
            ]
            name = "%s/%s" % phase
            phase_summary[name] = {
                "requests": len(records), "http_200": sum(r.get("status") == 200 for r in records),
                "errors": sum(bool(r.get("error")) for r in records),
                "content_checks_passed": sum(r.get("content_check", {}).get("passed") for r in records),
                "completion_tokens": {
                    "median": statistics.median(completion_tokens) if completion_tokens else None,
                    "min": min(completion_tokens) if completion_tokens else None,
                    "max": max(completion_tokens) if completion_tokens else None,
                },
                "latency_s": {"median": statistics.median(elapsed) if elapsed else None,
                              "p95": percentile(elapsed, .95), "max": max(elapsed) if elapsed else None},
            }
        errors = [record for record in summary_records if record.get("kind") == "request" and record.get("error")]
        recoveries = [record for record in summary_records if record.get("kind") == "recovery"]
        summary = {"kind": "summary", "utc": utc_now(), "output": os.path.abspath(output),
                   "duration_seconds": args.duration_seconds, "requests": len(requests),
                   "elapsed_seconds": time.monotonic() - started,
                   "completed_requested_duration": not stopped_on_transport_error,
                   "stopped_on_transport_error": stopped_on_transport_error,
                   "final_recovery_timeout_seconds": final_recovery_timeout,
                   "phase_summary": phase_summary, "record_errors": len(errors),
                   "recovery_passed": all(r["health_passed"] and r["marker_passed"] for r in recoveries),
                   "timeout_probe_outcome": (
                       None if timeout_probe_record is None else timeout_probe_record.get("outcome")
                   ),
                   "cancel_stream_observed": (
                       None if cancel_stream_record is None else cancel_stream_record.get("canceled_by_client")
                   )}
        log.write(summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        has_stress_failure = bool(errors) or any(
            record.get("status") != 200 or not record.get("content_check", {}).get("passed")
            for record in requests
        ) or not summary["recovery_passed"] or (
            not args.skip_probes
            and (
                not summary["cancel_stream_observed"]
                or summary["timeout_probe_outcome"] == "error"
            )
        )
        if args.fail_on_errors and has_stress_failure:
            return 1
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
