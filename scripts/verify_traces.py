"""Reads a trace back from LangSmith and verifies the required metadata keys."""

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
            root_run = client.read_run(run_id_part)
            break
        except Exception:
            if attempt == 4:
                print(f"  FAIL: could not read root run {run_id_part} after 5 attempts")
                return False
            time.sleep(2)

    meta = root_run.extra.get("metadata", {}) if root_run.extra else {}
    ok = True

    for key, expected in [
        ("prompt_name", expected_name),
        ("prompt_version", expected_version),
        ("environment", expected_env),
    ]:
        actual = meta.get(key)
        if actual == expected:
            print(f"  OK: root metadata {key} = {actual}")
        else:
            print(f"  FAIL: root metadata {key} expected={expected} actual={actual}")
            ok = False

    # Check child LLM run for LangSmith tracking metadata
    all_runs = list(client.list_runs(trace_id=root_run.trace_id))
    llm_run = None
    for r in all_runs:
        if r.run_type == "llm":
            llm_run = r
            break

    if llm_run:
        llm_meta = llm_run.extra.get("metadata", {}) if llm_run.extra else {}
        for key in ["ls_model_name", "ls_provider", "ls_temperature"]:
            if key in llm_meta:
                print(f"  OK: LLM metadata {key} = {llm_meta[key]}")
            else:
                print(f"  FAIL: LLM metadata {key} not found")
                ok = False
    else:
        print("  FAIL: could not find an LLM child run to check ls_ metadata")
        ok = False

    return ok


def call_invoke(base: str, prompt_name: str, version: str, env: str | None = None):
    body = {
        "input_text": "The World Health Organization declared on 5 May 2023 that COVID-19 is no longer a public health emergency of international concern.",
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
        ("extract_entities", "v2", "production", "production"),
    ]

    for prompt_name, version, send_env, expect_env in cases:
        label = f"{prompt_name}/{version} env={send_env or '(default)'}"
        print(f"\n--- {label} ---")
        try:
            resp = call_invoke(base, prompt_name, version, send_env)
        except Exception as exc:
            print(f"  FAIL: invoke error: {exc}")
            all_ok = False
            continue

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
