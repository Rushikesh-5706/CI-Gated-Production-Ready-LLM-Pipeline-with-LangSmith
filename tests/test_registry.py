import json
from pathlib import Path

import pytest

from app import registry


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


class TestRegistryValidation:
    def test_missing_template_token(self, tmp_path, monkeypatch):
        _write_prompt(
            tmp_path,
            "p",
            "v1",
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
        data = _valid_prompt()
        data["extra_key"] = "bad"
        _write_prompt(tmp_path, "p", "v1", data)
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="unexpected keys"):
            registry.load_all()

    def test_temperature_out_of_range(self, tmp_path, monkeypatch):
        _write_prompt(tmp_path, "p", "v1", _valid_prompt(model_parameters={"temperature": 3.0}))
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="temperature"):
            registry.load_all()


class TestVersionOrdering:
    def test_numeric_sort(self, tmp_path, monkeypatch):
        for v in ["v1", "v2", "v10", "v3"]:
            _write_prompt(tmp_path, "p", v, _valid_prompt())
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        registry.load_all()
        assert registry.list_prompts()["p"] == ["v1", "v2", "v3", "v10"]
        assert registry.latest_version("p") == "v10"


class TestRendering:
    def test_braces_in_input(self, tmp_path, monkeypatch):
        _write_prompt(tmp_path, "p", "v1", _valid_prompt(template="Process: {input_text}"))
        monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
        registry.load_all()
        pt = registry.get("p", "v1")
        result = registry.render(pt, 'text with {curly} and {"json": true}')
        assert "{curly}" in result
        assert '{"json": true}' in result
