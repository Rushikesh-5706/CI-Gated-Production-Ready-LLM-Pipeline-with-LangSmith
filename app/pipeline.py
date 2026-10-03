import logging
from dataclasses import dataclass

from langsmith import get_current_run_tree, traceable

from app import config, registry
from app.llm import LLMResult, chat_completion
from app.parsing import parse_entities, parse_summary

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output: str | dict
    trace_url: str
    llm_result: LLMResult


@traceable(name="render_prompt", run_type="chain")
def _render(template: registry.PromptTemplate, input_text: str) -> str:
    return registry.render(template, input_text)


@traceable(name="parse_output", run_type="chain")
def _parse(prompt_name: str, raw_text: str) -> str | dict:
    if prompt_name == "extract_entities":
        return parse_entities(raw_text)
    return parse_summary(raw_text)


def execute(
    prompt_name: str,
    prompt_version: str,
    input_text: str,
) -> PipelineResult:
    # registry.get raises PromptNotFoundError or VersionNotFoundError if absent.
    pt = registry.get(prompt_name, prompt_version)

    rendered = _render(pt, input_text)
    max_tokens = config.PROMPT_MAX_TOKENS.get(prompt_name, config.DEFAULT_MAX_TOKENS)

    response_format: dict | None = None
    if prompt_name == "extract_entities":
        response_format = {"type": "json_object"}

    llm_out = chat_completion(
        prompt=rendered,
        model=config.LLM_MODEL,
        temperature=pt.temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    llm_result: LLMResult = llm_out["result"]

    output = _parse(prompt_name, llm_result.text)

    return PipelineResult(output=output, trace_url="", llm_result=llm_result)


@traceable(name="invoke_pipeline", run_type="chain")
def invoke(
    prompt_name: str,
    prompt_version: str,
    input_text: str,
    environment: str,
) -> PipelineResult:
    rt = get_current_run_tree()
    metadata = {
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "environment": environment,
    }
    if rt:
        rt.add_metadata(metadata)
        rt.add_tags([prompt_name, prompt_version, environment])

    result = execute(prompt_name, prompt_version, input_text)

    trace_url = ""
    if rt:
        try:
            trace_url = rt.get_url()
        except Exception:
            log.warning("tracing disabled; trace_url will be empty")

    result.trace_url = trace_url
    return result
