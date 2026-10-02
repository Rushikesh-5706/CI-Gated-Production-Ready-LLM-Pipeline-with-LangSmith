# Rates from https://groq.com/pricing and https://console.groq.com/docs/models (October 2026)
# USD per million tokens

RATES: dict[str, dict[str, float]] = {
    "qwen/qwen3.8-27b": {
        "input": 0.80,
        "output": 4.00,
    },
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.60,
    },
    "openai/gpt-oss-20b": {
        "input": 0.075,
        "output": 0.30,
    },
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = RATES.get(model)
    if not rates:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost


def estimate_total(usage_by_model: dict[str, dict[str, int]]) -> dict[str, float]:
    costs: dict[str, float] = {}
    total = 0.0
    for model, counts in usage_by_model.items():
        cost = estimate_cost(model, counts["prompt_tokens"], counts["completion_tokens"])
        costs[model] = cost
        total += cost
    costs["total"] = total
    return costs
