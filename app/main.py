import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, registry
from app.errors import (
    LLMEmptyResponseError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
    PromptNotFoundError,
    VersionNotFoundError,
)
from app.pipeline import invoke
from app.schemas import HealthResponse, InvokeRequest, InvokeResponse, PromptInfo

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def _setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
        handlers=[handler],
        force=True,
    )


log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _setup_logging()
    registry.load_all()
    yield


app = FastAPI(title="CI-Gated LLM Pipeline", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    token = _request_id.set(req_id)
    t0 = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error in request")
        raise
    finally:
        elapsed = time.monotonic() - t0
        log.info(
            "%s %s %d %.3fs",
            request.method,
            request.url.path,
            response.status_code if "response" in dir() else 500,
            elapsed,
        )
        _request_id.reset(token)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    tracing_on = (
        config.LANGCHAIN_TRACING_V2.lower() == "true" and bool(config.LANGCHAIN_API_KEY)
    )
    return HealthResponse(
        status="ok",
        prompts_loaded=registry.count(),
        llm_configured=bool(config.GROQ_API_KEY),
        tracing_enabled=tracing_on,
    )


@app.get("/prompts", response_model=list[PromptInfo])
def list_prompts() -> list[PromptInfo]:
    result = []
    for name, versions in registry.list_prompts().items():
        changelogs = {}
        for v in versions:
            changelogs[v] = registry.get_changelog(name, v)
        result.append(PromptInfo(name=name, versions=versions, changelogs=changelogs))
    return result


@app.post("/invoke/{prompt_name}", response_model=InvokeResponse)
def invoke_prompt(prompt_name: str, body: InvokeRequest) -> InvokeResponse:
    environment = body.environment or config.DEFAULT_ENVIRONMENT
    log.info(
        "invoke prompt=%s version=%s env=%s input_len=%d",
        prompt_name,
        body.prompt_version,
        environment,
        len(body.input_text),
    )

    result = invoke(
        prompt_name=prompt_name,
        prompt_version=body.prompt_version,
        input_text=body.input_text,
        environment=environment,
    )

    return InvokeResponse(output=result.output, trace_url=result.trace_url)


@app.exception_handler(PromptNotFoundError)
async def handle_prompt_not_found(request: Request, exc: PromptNotFoundError) -> JSONResponse:
    log.info("prompt not found: %s", exc)
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(VersionNotFoundError)
async def handle_version_not_found(request: Request, exc: VersionNotFoundError) -> JSONResponse:
    log.info("version not found: %s", exc)
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(LLMNotConfiguredError)
async def handle_llm_not_configured(
    request: Request, exc: LLMNotConfiguredError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMRateLimitError)
async def handle_rate_limit(request: Request, exc: LLMRateLimitError) -> JSONResponse:
    headers = {}
    if exc.retry_after is not None:
        headers["Retry-After"] = str(int(exc.retry_after))
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
        headers=headers,
    )


@app.exception_handler(LLMTimeoutError)
async def handle_timeout(request: Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": "upstream LLM request timed out"})


@app.exception_handler(LLMEmptyResponseError)
async def handle_empty_response(request: Request, exc: LLMEmptyResponseError) -> JSONResponse:
    log.warning("empty response from model: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "model returned empty response"})


@app.exception_handler(LLMUpstreamError)
async def handle_upstream(request: Request, exc: LLMUpstreamError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "upstream LLM error"})


@app.exception_handler(Exception)
async def handle_generic(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
