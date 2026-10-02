"""Reads a trace back from LangSmith and verifies the three required metadata keys."""

import sys
import time

import requests as req
from langsmith import Client


def verify_trace(trace_url: str, expected_name: str, expected_version: str, expected_env: str):
    if "/trace/" not in trace_url:
        print(f"  FAIL: trace_url does not contain /trace/: {trace_url}")
        return False

    parts = trace_url.split("/trace/")
    run_id_part = parts[1].split("/")[0]

    client = Client()

    # The trace may not be flushed yet
    for attempt in range(5):
        try:
            run = client.read_run(run_id_part)
            break
        except Exception:
            if attempt == 4:
                print(f"  FAIL: could not read run {run_id_part} after 5 attempts")
                return False
            time.sleep(2)

    meta = run.extra.get("metadata", {}) if run.extra else {}
    ok = True

    for key, expected in [
        ("prompt_name", expected_name),
        ("prompt_version", expected_version),
        ("environment", expected_env),
    ]:
        actual = meta.get(key)
        if actual == expected:
            print(f"  OK: {key} = {actual}")
        else:
            print(f"  FAIL: {key} expected={expected} actual={actual}")
            ok = False

    if run.total_tokens is not None and run.total_tokens > 0:
        print(f"  OK: total_tokens = {run.total_tokens}")
    else:
        print(f"  INFO: total_tokens = {run.total_tokens} (may not be set on root)")

    if run.latency is not None:
        print(f"  OK: latency = {run.latency}")

    return ok


def call_invoke(base: str, prompt_name: str, version: str, env: str | None = None):
    body = {
        "input_text": "The World Health Organization declared on 5 May 2023 that COVID-19 is no longer a public health emergency of international concern. Director-General Tedros Adhanom Ghebreyesus made the announcement from Geneva after a meeting of the emergency committee.",
        "prompt_version": version,
    }
    if env:
        body["environment"] = env

    r = req.post(f"{base}/invoke/{prompt_name}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    base = "http://127.0.0.1:8000"
    all_ok = True

    cases = [
        ("summarize_text", "v1", None, "staging"),
        ("summarize_text", "v2", "production", "production"),
        ("extract_entities", "v2", None, "staging"),
        ("extract_entities", "v1", "production", "production"),
    ]

    for prompt_name, version, send_env, expect_env in cases:
        label = f"{prompt_name}/{version} env={send_env or '(default)'}"
        print(f"\n--- {label} ---")
        resp = call_invoke(base, prompt_name, version, send_env)
        trace_url = resp.get("trace_url", "")
        print(f"  trace_url: {trace_url[:80]}...")

        time.sleep(3)

        ok = verify_trace(trace_url, prompt_name, version, expect_env)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("ALL TRACE VERIFICATIONS PASSED")
    else:
        print("SOME TRACE VERIFICATIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
