import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, registry
from app.errors import (
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
    PromptNotFoundError,
    VersionNotFoundError,
)
from app.pipeline import invoke
from app.schemas import HealthResponse, InvokeRequest, InvokeResponse, PromptInfo

log = logging.getLogger(__name__)

app = FastAPI(title="CI-Gated LLM Pipeline")


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    registry.load_all()


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed = time.monotonic() - t0
    log.info(
        "%s %s %d %.3fs req=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        request_id,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        prompts_loaded=registry.count(),
        llm_configured=bool(config.GROQ_API_KEY),
        tracing_enabled=config.LANGCHAIN_TRACING_V2.lower() == "true",
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
    input_len = len(body.input_text)
    log.info(
        "invoke prompt=%s version=%s env=%s input_len=%d",
        prompt_name,
        body.prompt_version,
        environment,
        input_len,
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
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
    )


@app.exception_handler(VersionNotFoundError)
async def handle_version_not_found(request: Request, exc: VersionNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
    )


@app.exception_handler(LLMNotConfiguredError)
async def handle_llm_not_configured(request: Request, exc: LLMNotConfiguredError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
    )


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
    return JSONResponse(
        status_code=504,
        content={"detail": "upstream LLM request timed out"},
    )


@app.exception_handler(LLMUpstreamError)
async def handle_upstream(request: Request, exc: LLMUpstreamError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"detail": "upstream LLM error"},
    )


@app.exception_handler(Exception)
async def handle_generic(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )
