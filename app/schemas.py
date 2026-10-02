import re

from pydantic import BaseModel, Field, field_validator


class InvokeRequest(BaseModel, extra="forbid"):
    input_text: str = Field(..., min_length=1, max_length=20000)
    prompt_version: str = Field(..., pattern=r"^v[0-9]+$")
    environment: str | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("input_text", mode="before")
    @classmethod
    def strip_input(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.match(r"^[a-z0-9_-]{1,32}$", v):
            raise ValueError("environment must match [a-z0-9_-] and be 1-32 chars")
        return v


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
