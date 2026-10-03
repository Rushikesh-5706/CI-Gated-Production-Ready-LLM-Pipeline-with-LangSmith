"""LLM token cost estimation.

Rates sourced from https://console.groq.com/docs/models and https://groq.com/pricing
as of 2026-10-03.  Costs are in USD per 1 million tokens.
"""

# Model name must match LLM_MODEL and JUDGE_MODEL in app/config.py,
# docker-compose.yml and .env.example exactly.
_PRICING: dict[str, tuple[float, float]] = {
    # (input_cost_per_1M, output_cost_per_1M)
    # Source: https://groq.com/pricing (2026-10-03)
    "qwen/qwen3.8-27b": (0.10, 0.20),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the estimated cost in USD.

    Raises KeyError with a clear message if the model has no pricing entry.
    """
    if model not in _PRICING:
        raise KeyError(
            f"no pricing entry for model {model!r}; add it to evaluation/pricing.py"
        )
    in_rate, out_rate = _PRICING[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def estimate_total(usage_by_model: dict[str, dict[str, int]]) -> dict[str, float]:
    """Compute per-model and total costs.

    Unknown models are assigned 0.0 cost and logged as 'unpriced'.
    """
    costs: dict[str, float] = {}
    total = 0.0
    for model, counts in usage_by_model.items():
        try:
            cost = estimate_cost(
                model,
                counts.get("prompt_tokens", 0),
                counts.get("completion_tokens", 0),
            )
        except KeyError:
            cost = 0.0
            costs[f"{model}_unpriced"] = 0.0
        costs[model] = cost
        total += cost
    costs["total"] = total
    return costs


def format_cost(cost: float) -> str:
    return f"${cost:.4f}"


def model_is_priced(model: str) -> bool:
    return model in _PRICING
