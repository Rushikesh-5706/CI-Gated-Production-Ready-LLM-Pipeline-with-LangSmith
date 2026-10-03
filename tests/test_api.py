"""API integration tests.

These tests run the real route → schema → pipeline → parser path.
The only thing patched is the Groq HTTP client so no real LLM calls occur.
A separate set of pure routing tests uses a full invoke mock for speed.
"""

import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_groq_response(text: str, model: str = "qwen/qwen3.8-27b") -> MagicMock:
    usage = types.SimpleNamespace(
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    )
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content=text),
        finish_reason="stop",
    )
    return types.SimpleNamespace(choices=[choice], usage=usage, model=model)


def _groq_patch(text: str):
    """Return a context manager that patches the Groq completions endpoint."""
    return patch(
        "app.llm._get_client",
        return_value=MagicMock(
            chat=MagicMock(
                completions=MagicMock(
                    create=MagicMock(return_value=_make_groq_response(text))
                )
            )
        ),
    )


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["prompts_loaded"] >= 4
        assert isinstance(body["llm_configured"], bool)
        assert isinstance(body["tracing_enabled"], bool)

    def test_health_llm_configured_false_without_key(self):
        with patch("app.config.GROQ_API_KEY", ""), patch("app.config.LANGCHAIN_API_KEY", ""):
            from app.main import app

            with TestClient(app) as c:
                r = c.get("/health")
                assert r.status_code == 200
                assert r.json()["llm_configured"] is False
                assert r.json()["tracing_enabled"] is False


class TestPromptsEndpoint:
    def test_prompts_lists_known_prompts(self, client):
        r = client.get("/prompts")
        assert r.status_code == 200
        names = {p["name"] for p in r.json()}
        assert "summarize_text" in names
        assert "extract_entities" in names

    def test_prompts_includes_versions(self, client):
        r = client.get("/prompts")
        for p in r.json():
            assert len(p["versions"]) >= 2


class TestInvokeEndpoint:
    def test_summarize_returns_string(self, client):
        with _groq_patch("A brief summary of the text."):
            r = client.post(
                "/invoke/summarize_text",
                json={"input_text": "Some text to summarize.", "prompt_version": "v1"},
            )
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["output"], str)
        assert "trace_url" in body

    def test_summarize_v2_works(self, client):
        with _groq_patch("Another summary."):
            r = client.post(
                "/invoke/summarize_text",
                json={"input_text": "Some text.", "prompt_version": "v2"},
            )
        assert r.status_code == 200

    def test_extract_returns_dict(self, client):
        payload = '{"names": ["Alice"], "dates": [], "locations": ["Paris"]}'
        with _groq_patch(payload):
            r = client.post(
                "/invoke/extract_entities",
                json={"input_text": "Alice went to Paris.", "prompt_version": "v1"},
            )
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["output"], dict)
        assert "names" in body["output"]

    def test_environment_default_staging(self, client):
        with _groq_patch("Summary output here."):
            r = client.post(
                "/invoke/summarize_text",
                json={"input_text": "text", "prompt_version": "v1"},
            )
        assert r.status_code == 200

    def test_environment_explicit_production(self, client):
        with _groq_patch("Summary."):
            r = client.post(
                "/invoke/summarize_text",
                json={
                    "input_text": "text",
                    "prompt_version": "v1",
                    "environment": "production",
                },
            )
        assert r.status_code == 200

    def test_unknown_prompt_404(self, client):
        with _groq_patch("x"):
            r = client.post(
                "/invoke/nonexistent_prompt",
                json={"input_text": "test", "prompt_version": "v1"},
            )
        assert r.status_code == 404

    def test_unknown_version_404(self, client):
        with _groq_patch("x"):
            r = client.post(
                "/invoke/summarize_text",
                json={"input_text": "test", "prompt_version": "v99"},
            )
        assert r.status_code == 404

    def test_missing_key_503(self):
        with patch("app.config.GROQ_API_KEY", ""):
            from app.main import app

            with TestClient(app) as c:
                r = c.post(
                    "/invoke/summarize_text",
                    json={"input_text": "test", "prompt_version": "v1"},
                )
        assert r.status_code == 503

    def test_invalid_version_format_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={"input_text": "test", "prompt_version": "version1"},
        )
        assert r.status_code == 422

    def test_extra_field_rejected(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={"input_text": "test", "prompt_version": "v1", "extra": "bad"},
        )
        assert r.status_code == 422

    def test_environment_integer_yields_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={"input_text": "test", "prompt_version": "v1", "environment": 5},
        )
        assert r.status_code == 422

    def test_environment_list_yields_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={"input_text": "test", "prompt_version": "v1", "environment": ["a"]},
        )
        assert r.status_code == 422

    def test_environment_null_uses_default(self, client):
        with _groq_patch("Summary."):
            r = client.post(
                "/invoke/summarize_text",
                json={
                    "input_text": "test",
                    "prompt_version": "v1",
                    "environment": None,
                },
            )
        assert r.status_code == 200

    def test_environment_uppercase_yields_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={"input_text": "test", "prompt_version": "v1", "environment": "Staging"},
        )
        assert r.status_code == 422

    def test_environment_too_long_yields_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={
                "input_text": "test",
                "prompt_version": "v1",
                "environment": "a" * 33,
            },
        )
        assert r.status_code == 422

    def test_empty_input_text_yields_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={"input_text": "", "prompt_version": "v1"},
        )
        assert r.status_code == 422

    def test_missing_prompt_version_yields_422(self, client):
        r = client.post(
            "/invoke/summarize_text",
            json={"input_text": "test"},
        )
        assert r.status_code == 422

    def test_empty_response_yields_502(self, client):
        with _groq_patch(""):
            r = client.post(
                "/invoke/summarize_text",
                json={"input_text": "test", "prompt_version": "v1"},
            )
        assert r.status_code == 502

    def test_timeout_yields_504(self, client):
        import groq as groq_mod

        with patch("app.llm._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = (
                groq_mod.APITimeoutError(request=MagicMock())
            )
            with patch("app.config.LLM_MAX_RETRIES", 0):
                r = client.post(
                    "/invoke/summarize_text",
                    json={"input_text": "test", "prompt_version": "v1"},
                )
        assert r.status_code == 504
