import logging
import os

log = logging.getLogger(__name__)

PORT: int = int(os.environ.get("PORT", "8000"))
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
DEFAULT_ENVIRONMENT: str = os.environ.get("DEFAULT_ENVIRONMENT", "staging")

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
JUDGE_MODEL: str = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")
LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
LLM_REQUESTS_PER_MINUTE: int = int(os.environ.get("LLM_REQUESTS_PER_MINUTE", "25"))
LLM_TOKENS_PER_MINUTE: int = int(os.environ.get("LLM_TOKENS_PER_MINUTE", "5000"))
LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "6"))

LANGCHAIN_API_KEY: str = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2: str = os.environ.get("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT: str = os.environ.get("LANGCHAIN_PROJECT", "ci-gated-llm-pipeline")

EVAL_ENVIRONMENT: str = os.environ.get("EVAL_ENVIRONMENT", "ci")
EVAL_SAMPLE_SIZE: int = int(os.environ.get("EVAL_SAMPLE_SIZE", "0"))
EVAL_MAX_CONCURRENCY: int = int(os.environ.get("EVAL_MAX_CONCURRENCY", "3"))

PROMPT_MAX_TOKENS: dict[str, int] = {
    "summarize_text": 256,
    "extract_entities": 512,
}
DEFAULT_MAX_TOKENS: int = 512


def _mirror_langchain_to_langsmith() -> None:
    """Copy LANGCHAIN_* env vars to their LANGSMITH_* counterparts when missing."""
    mapping = {
        "LANGCHAIN_API_KEY": "LANGSMITH_API_KEY",
        "LANGCHAIN_TRACING_V2": "LANGSMITH_TRACING_V2",
        "LANGCHAIN_PROJECT": "LANGSMITH_PROJECT",
    }
    for src, dst in mapping.items():
        val = os.environ.get(src, "")
        if val and not os.environ.get(dst, ""):
            os.environ[dst] = val


_mirror_langchain_to_langsmith()

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set; LLM calls will fail with 503")
