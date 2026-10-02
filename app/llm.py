import logging
import random
import time
from dataclasses import dataclass

import groq
from langsmith import traceable

from app import config
from app.limiter import get_limiter

log = logging.getLogger(__name__)

_client: groq.Groq | None = None
_usage_callback = None


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


def set_usage_callback(cb) -> None:
    global _usage_callback
    _usage_callback = cb


def _get_client() -> groq.Groq:
    global _client
    if _client is None:
        _client = groq.Groq(
            api_key=config.GROQ_API_KEY,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _client


def _estimate_tokens(prompt: str, max_tokens: int) -> int:
    return len(prompt) // 4 + max_tokens


@traceable(run_type="llm", name="groq_chat_completion")
def chat_completion(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: dict | None = None,
) -> dict:
    """Single LLM call with rate limiting and retries. Returns a dict for LangSmith tracing."""
    from app.errors import (
        LLMNotConfiguredError,
        LLMRateLimitError,
        LLMTimeoutError,
        LLMUpstreamError,
    )

    if not config.GROQ_API_KEY:
        raise LLMNotConfiguredError("LLM provider is not configured")

    client = _get_client()
    limiter = get_limiter(model, config.LLM_REQUESTS_PER_MINUTE, config.LLM_TOKENS_PER_MINUTE)
    estimated = _estimate_tokens(prompt, max_tokens)
    limiter.wait_for_capacity(estimated)

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    last_exc = None
    for attempt in range(1 + config.LLM_MAX_RETRIES):
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.monotonic() - t0
            usage = resp.usage
            prompt_tok = usage.prompt_tokens if usage else 0
            comp_tok = usage.completion_tokens if usage else 0
            total_tok = usage.total_tokens if usage else 0

            limiter.reconcile_tokens(estimated, total_tok)

            text = resp.choices[0].message.content or ""

            if _usage_callback:
                _usage_callback(model, prompt_tok, comp_tok)

            return {
                "result": LLMResult(
                    text=text,
                    model=model,
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                    latency_seconds=round(elapsed, 3),
                ),
                "usage_metadata": {
                    "input_tokens": prompt_tok,
                    "output_tokens": comp_tok,
                    "total_tokens": total_tok,
                },
                "ls_provider": "groq",
                "ls_model_name": model,
                "ls_temperature": temperature,
            }

        except groq.RateLimitError as exc:
            last_exc = exc
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after_str = exc.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
            if attempt == config.LLM_MAX_RETRIES:
                raise LLMRateLimitError(str(exc), retry_after=retry_after) from exc
            sleep_time = retry_after or _backoff(attempt)
            log.warning("rate limited, sleeping %.1fs (attempt %d)", sleep_time, attempt + 1)
            time.sleep(sleep_time)

        except groq.APITimeoutError as exc:
            last_exc = exc
            if attempt == config.LLM_MAX_RETRIES:
                raise LLMTimeoutError(str(exc)) from exc
            time.sleep(_backoff(attempt))

        except groq.APIConnectionError as exc:
            last_exc = exc
            if attempt == config.LLM_MAX_RETRIES:
                raise LLMUpstreamError(str(exc)) from exc
            time.sleep(_backoff(attempt))

        except groq.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                if attempt == config.LLM_MAX_RETRIES:
                    raise LLMUpstreamError(str(exc)) from exc
                time.sleep(_backoff(attempt))
            else:
                raise LLMUpstreamError(str(exc)) from exc

    raise LLMUpstreamError(str(last_exc))


def _backoff(attempt: int) -> float:
    base = min(2**attempt, 30)
    return base + random.uniform(0, base * 0.5)
