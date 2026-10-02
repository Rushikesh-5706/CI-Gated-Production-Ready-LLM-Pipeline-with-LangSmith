from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _mock_invoke(prompt_name, prompt_version, input_text, environment, **kwargs):
    from app import registry
    from app.errors import PromptNotFoundError, VersionNotFoundError
    from app.llm import LLMResult
    from app.pipeline import PipelineResult

    names = registry.prompt_names()
    if prompt_name not in names:
        raise PromptNotFoundError(
            f"prompt '{prompt_name}' not found; available: {names}",
            available=names,
        )
    pt = registry.get(prompt_name, prompt_version)
    if pt is None:
        versions = registry.list_prompts().get(prompt_name, [])
        raise VersionNotFoundError(
            f"version '{prompt_version}' not found for '{prompt_name}'; available: {versions}",
            available=versions,
        )

    lr = LLMResult(
        text="mocked", model="test", prompt_tokens=10, completion_tokens=10, latency_seconds=0.1
    )
    output = "mocked output"
    if prompt_name == "extract_entities":
        output = {"names": [], "dates": [], "locations": []}
    return PipelineResult(output=output, trace_url="https://example.com/trace", llm_result=lr)


@pytest.fixture
def client():
    with patch("app.main.invoke", side_effect=_mock_invoke):
        from app.main import app

        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert isinstance(data["prompts_loaded"], int)


class TestInvokeEndpoint:
    def test_summarize_text_v1(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={
                "input_text": "Some text to summarize.",
                "prompt_version": "v1",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "output" in data
        assert "trace_url" in data

    def test_unknown_prompt_404(self, client):
        r = client.post(
            "/invoke/nonexistent_prompt",
            json={
                "input_text": "test",
                "prompt_version": "v1",
            },
        )
        assert r.status_code == 404
        assert "available" in r.json()["detail"]

    def test_unknown_version_404(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={
                "input_text": "test",
                "prompt_version": "v99",
            },
        )
        assert r.status_code == 404
        assert "available" in r.json()["detail"]

    def test_empty_input_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={
                "input_text": "",
                "prompt_version": "v1",
            },
        )
        assert r.status_code == 422

    def test_missing_version_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={
                "input_text": "test",
            },
        )
        assert r.status_code == 422

    def test_invalid_version_format_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={
                "input_text": "test",
                "prompt_version": "version1",
            },
        )
        assert r.status_code == 422

    def test_extra_field_rejected(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={
                "input_text": "test",
                "prompt_version": "v1",
                "extra": "bad",
            },
        )
        assert r.status_code == 422


class TestLLMNotConfigured:
    def test_503_when_key_missing(self):
        from app.errors import LLMNotConfiguredError

        def raise_not_configured(*args, **kwargs):
            raise LLMNotConfiguredError("LLM provider is not configured")

        with patch("app.main.invoke", side_effect=raise_not_configured):
            from app.main import app

            with TestClient(app) as c:
                r = c.post(
                    "/invoke/summarize_text",
                    json={
                        "input_text": "test",
                        "prompt_version": "v1",
                    },
                )
                assert r.status_code == 503
