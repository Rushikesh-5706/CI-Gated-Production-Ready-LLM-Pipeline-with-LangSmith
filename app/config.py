"""Application settings loaded from environment variables.

Call ``load_dotenv`` first so a local ``.env`` file is respected without
requiring callers to export variables manually.  Real environment variables
always win (``override=False``).
"""

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

log = logging.getLogger(__name__)

PORT: int = int(os.environ.get("PORT", "8000"))
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
DEFAULT_ENVIRONMENT: str = os.environ.get("DEFAULT_ENVIRONMENT", "staging")

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
# qwen/qwen3.8-27b: no reasoning bleed at default temperature, fast, low cost.
# Probe result (2026-10-03): finish_reason=stop 3/3, empty=0, think=0.
LLM_MODEL: str = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
# Same model used as judge with json_object response_format.
# Probe result: finish_reason=stop 3/3, avg_comp_tok=85, avg_lat=0.67s.
JUDGE_MODEL: str = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")

LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
LLM_REQUESTS_PER_MINUTE: int = int(os.environ.get("LLM_REQUESTS_PER_MINUTE", "25"))
LLM_TOKENS_PER_MINUTE: int = int(os.environ.get("LLM_TOKENS_PER_MINUTE", "5000"))
LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "6"))

LANGCHAIN_API_KEY: str = os.environ.get("LANGCHAIN_API_KEY", "")
# Default to true so tracing works after copying .env.example.
LANGCHAIN_TRACING_V2: str = os.environ.get("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT: str = os.environ.get("LANGCHAIN_PROJECT", "ci-gated-llm-pipeline")

EVAL_ENVIRONMENT: str = os.environ.get("EVAL_ENVIRONMENT", "ci")
EVAL_SAMPLE_SIZE: int = int(os.environ.get("EVAL_SAMPLE_SIZE", "0"))
EVAL_MAX_CONCURRENCY: int = int(os.environ.get("EVAL_MAX_CONCURRENCY", "3"))

# Per-prompt token budgets.  Judge uses JUDGE_MAX_TOKENS.
PROMPT_MAX_TOKENS: dict[str, int] = {
    "summarize_text": 256,
    "extract_entities": 512,
}
JUDGE_MAX_TOKENS: int = int(os.environ.get("JUDGE_MAX_TOKENS", "512"))
DEFAULT_MAX_TOKENS: int = 512

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set; LLM calls will fail with 503")
