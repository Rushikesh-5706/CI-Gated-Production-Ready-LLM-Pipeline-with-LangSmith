import json
from pathlib import Path

import pytest

from app import registry
from app.errors import PromptNotFoundError, VersionNotFoundError


@pytest.fixture(autouse=True)
def _clean():
    registry._cache.clear()
    registry._versions.clear()
    yield
    registry._cache.clear()
    registry._versions.clear()


def _write_prompt(base: Path, name: str, version: str, data: dict):
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{version}.json").write_text(json.dumps(data))


def _valid_prompt(**overrides):
    default = {
        "template": "Do something with {input_text}",
        "changelog": "test version",
        "model_parameters": {"temperature": 0.5},
    }
    default.update(overrides)
    return default


def _make_required(base: Path):
    """Write minimal required prompts so load_all() passes the startup check."""
    for name in ("summarize_text", "extract_entities"):
        _write_prompt(base, name, "v1", _valid_prompt())


class TestRegistryValidation:
    def test_missing_template_token(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        _write_prompt(
            tmp_path,
            "summarize_text",
            "v2",
            {
                "template": "No placeholder here",
                "changelog": "test",
                "model_parameters": {"temperature": 0.5},
            },
        )
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="input_text"):
            registry.load_all()

    def test_extra_keys_rejected(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        data = _valid_prompt()
        data["extra_key"] = "bad"
        _write_prompt(tmp_path, "summarize_text", "v2", data)
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="unexpected keys"):
            registry.load_all()

    def test_temperature_out_of_range(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        _write_prompt(
            tmp_path,
            "summarize_text",
            "v2",
            _valid_prompt(model_parameters={"temperature": 3.0}),
        )
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="temperature"):
            registry.load_all()

    def test_missing_temperature_required(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        _write_prompt(
            tmp_path,
            "summarize_text",
            "v2",
            {
                "template": "Summarize {input_text}",
                "changelog": "no temperature",
                "model_parameters": {},
            },
        )
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="temperature"):
            registry.load_all()

    def test_missing_model_parameters_required(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        _write_prompt(
            tmp_path,
            "summarize_text",
            "v2",
            {
                "template": "Summarize {input_text}",
                "changelog": "no params",
            },
        )
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="model_parameters"):
            registry.load_all()

    def test_missing_required_prompt_raises(self, tmp_path, monkeypatch):
        _write_prompt(tmp_path, "summarize_text", "v1", _valid_prompt())
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(RuntimeError, match="extract_entities"):
            registry.load_all()


class TestVersionOrdering:
    def test_numeric_sort(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        for v in ["v2", "v10", "v3"]:
            _write_prompt(tmp_path, "summarize_text", v, _valid_prompt())
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        registry.load_all()
        assert registry.list_prompts()["summarize_text"] == ["v1", "v2", "v3", "v10"]
        assert registry.latest_version("summarize_text") == "v10"


class TestRendering:
    def test_braces_in_input(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        registry.load_all()
        pt = registry.get("summarize_text", "v1")
        result = registry.render(pt, 'text with {curly} and {"json": true}')
        assert "{curly}" in result
        assert '{"json": true}' in result


class TestRegistryGet:
    def test_unknown_prompt_raises(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        registry.load_all()
        with pytest.raises(PromptNotFoundError):
            registry.get("nonexistent", "v1")

    def test_unknown_version_raises(self, tmp_path, monkeypatch):
        _make_required(tmp_path)
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        registry.load_all()
        with pytest.raises(VersionNotFoundError):
            registry.get("summarize_text", "v99")
