from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class InvokeRequest(BaseModel, extra="forbid"):
    input_text: str = Field(..., min_length=1, max_length=20000)
    prompt_version: str = Field(..., pattern=r"^v[0-9]+$")
    # Typed field with a regex constraint: Pydantic validates the type first,
    # so a non-string value (int, list, null) yields 422 before the pattern check.
    environment: Annotated[
        str,
        StringConstraints(pattern=r"^[a-z0-9_-]+$", min_length=1, max_length=32),
    ] | None = Field(default=None)

    @staticmethod
    def _strip(v: str) -> str:
        return v.strip()

    model_config = {"str_strip_whitespace": True}


class InvokeResponse(BaseModel):
    output: str | dict
    trace_url: str


class HealthResponse(BaseModel):
    status: str
    prompts_loaded: int
    llm_configured: bool
    tracing_enabled: bool


class PromptInfo(BaseModel):
    name: str
    versions: list[str]
    changelogs: dict[str, str]
