"""Probe Groq models for reasoning bleed, empty responses, and token budgets.

Usage:
    python scripts/probe_models.py

Prints a table of finish_reason counts, empty count, think-block count,
mean completion tokens, and mean latency for each case x model combination.
Results inform which models and max_tokens values are safe for the pipeline
and the LLM-as-judge evaluator.
"""

import os
import sys
import time

# Load .env so the script works after `cp .env.example .env`
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    sys.exit("groq package not installed; run: pip install groq")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    sys.exit("GROQ_API_KEY is not set")

CLIENT = Groq(api_key=GROQ_API_KEY)

PIPELINE_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")
PIPELINE_SUM_TOKENS = int(os.environ.get("PIPELINE_SUM_TOKENS", "256"))
PIPELINE_EXT_TOKENS = int(os.environ.get("PIPELINE_EXT_TOKENS", "512"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "512"))

MSG_SUM = [
    {
        "role": "user",
        "content": (
            "Summarize the following passage in 25 to 40 words:\n\n"
            "The Millbrook city council unanimously approved a revised municipal budget on Tuesday "
            "evening, allocating an additional 2.3 million dollars to road infrastructure repairs "
            "across twelve districts. The decision followed three months of public consultation."
        ),
    }
]
MSG_EXT = [
    {
        "role": "user",
        "content": (
            "Extract named entities from the text below. Return a JSON object with exactly three "
            "keys: names (list of person names without titles), dates (list of date strings as "
            "written), locations (list of place names as written). Text: "
            "Clara Nguyen and James Osei signed the accord in Helsinki on 22 September 2022."
        ),
    }
]
MSG_JUDGE = [
    {
        "role": "system",
        "content": (
            "You are evaluating the quality of a text summary. "
            "Respond only with a JSON object with keys "
            "relevance_score (integer 1-5), conciseness_score (integer 1-5), "
            "and justification (string)."
        ),
    },
    {
        "role": "user",
        "content": (
            "Original text: The Millbrook city council approved a revised municipal budget.\n"
            "Summary: The council approved the budget."
        ),
    },
]

CASES = [
    ("pipeline/sum", PIPELINE_MODEL, MSG_SUM, PIPELINE_SUM_TOKENS, {}),
    (
        "pipeline/ext",
        PIPELINE_MODEL,
        MSG_EXT,
        PIPELINE_EXT_TOKENS,
        {"response_format": {"type": "json_object"}},
    ),
    (
        "judge",
        JUDGE_MODEL,
        MSG_JUDGE,
        JUDGE_MAX_TOKENS,
        {"response_format": {"type": "json_object"}},
    ),
]

REPEATS = 3


def probe_once(model: str, messages: list, max_tokens: int, extra: dict) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        **extra,
    }
    t0 = time.monotonic()
    resp = CLIENT.chat.completions.create(**kwargs)
    lat = time.monotonic() - t0
    choice = resp.choices[0]
    content = choice.message.content or ""
    return {
        "finish_reason": choice.finish_reason,
        "empty": content == "",
        "has_think": "<think>" in content,
        "completion_tokens": resp.usage.completion_tokens,
        "latency": lat,
    }


def probe_case(label: str, model: str, messages: list, max_tokens: int, extra: dict) -> None:
    fr_counts: dict[str, int] = {}
    empty_count = 0
    think_count = 0
    tok_total = 0
    lat_total = 0.0
    for i in range(REPEATS):
        try:
            r = probe_once(model, messages, max_tokens, extra)
        except Exception as exc:
            print(f"  [{label}] attempt {i+1} error: {exc}")
            fr_counts["ERROR"] = fr_counts.get("ERROR", 0) + 1
            continue
        fr_counts[r["finish_reason"]] = fr_counts.get(r["finish_reason"], 0) + 1
        if r["empty"]:
            empty_count += 1
        if r["has_think"]:
            think_count += 1
        tok_total += r["completion_tokens"]
        lat_total += r["latency"]
        if i < REPEATS - 1:
            time.sleep(1.5)

    n = REPEATS - fr_counts.get("ERROR", 0)
    avg_tok = tok_total / n if n else 0
    avg_lat = lat_total / n if n else 0
    fr_str = str(fr_counts)
    verdict = "OK" if (empty_count == 0 and think_count == 0 and "ERROR" not in fr_counts) else "WARN"
    print(
        f"{label:<20} {model:<22} {fr_str:<28} "
        f"empty={empty_count}/{REPEATS} think={think_count}/{REPEATS} "
        f"avg_tok={avg_tok:>5.0f} avg_lat={avg_lat:>5.2f}s  [{verdict}]"
    )


def main() -> None:
    print(
        f"{'Case':<20} {'Model':<22} {'finish_reason counts':<28} "
        f"empty  think  avg_tok avg_lat  verdict"
    )
    print("-" * 120)
    for label, model, messages, max_tokens, extra in CASES:
        probe_case(label, model, messages, max_tokens, extra)
        time.sleep(2)

    print()
    print(f"Pipeline model : {PIPELINE_MODEL}  (pipeline_sum_tokens={PIPELINE_SUM_TOKENS}, pipeline_ext_tokens={PIPELINE_EXT_TOKENS})")
    print(f"Judge model    : {JUDGE_MODEL}  (judge_max_tokens={JUDGE_MAX_TOKENS})")


if __name__ == "__main__":
    main()
