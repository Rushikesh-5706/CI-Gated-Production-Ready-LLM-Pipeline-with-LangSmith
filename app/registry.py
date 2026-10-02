import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
VERSION_RE = re.compile(r"^v(\d+)$")
ALLOWED_TOP_KEYS = {"template", "changelog", "model_parameters"}


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str
    changelog: str
    temperature: float


_cache: dict[tuple[str, str], PromptTemplate] = {}
_versions: dict[str, list[str]] = {}


def _validate_and_load(path: Path, name: str, version: str) -> PromptTemplate:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a JSON object")

    extra = set(data.keys()) - ALLOWED_TOP_KEYS
    if extra:
        raise ValueError(f"{path}: unexpected keys {extra}")

    template = data.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"{path}: 'template' must be a non-empty string")
    if "{input_text}" not in template:
        raise ValueError(f"{path}: 'template' must contain '{{input_text}}'")

    changelog = data.get("changelog")
    if not isinstance(changelog, str) or not changelog.strip():
        raise ValueError(f"{path}: 'changelog' must be a non-empty string")

    params = data.get("model_parameters", {})
    if not isinstance(params, dict):
        raise ValueError(f"{path}: 'model_parameters' must be an object")

    temp = params.get("temperature", 0.7)
    if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
        raise ValueError(f"{path}: temperature must be a number between 0 and 2")

    return PromptTemplate(
        name=name,
        version=version,
        template=template,
        changelog=changelog,
        temperature=float(temp),
    )


def load_all() -> None:
    _cache.clear()
    _versions.clear()

    if not PROMPTS_DIR.is_dir():
        raise RuntimeError(f"prompts directory not found: {PROMPTS_DIR}")

    for prompt_dir in sorted(PROMPTS_DIR.iterdir()):
        if not prompt_dir.is_dir():
            continue
        name = prompt_dir.name
        vers: list[tuple[int, str]] = []

        for f in sorted(prompt_dir.iterdir()):
            if not f.is_file() or f.suffix != ".json":
                continue
            stem = f.stem
            m = VERSION_RE.match(stem)
            if not m:
                continue
            num = int(m.group(1))
            version = stem
            pt = _validate_and_load(f, name, version)
            _cache[(name, version)] = pt
            vers.append((num, version))

        vers.sort(key=lambda x: x[0])
        _versions[name] = [v for _, v in vers]

    total = len(_cache)
    log.info("loaded %d prompt templates across %d prompts", total, len(_versions))


def get(name: str, version: str) -> PromptTemplate:
    key = (name, version)
    if name not in _versions:
        return None  # type: ignore[return-value]
    if key not in _cache:
        return None  # type: ignore[return-value]
    return _cache[key]


def latest_version(name: str) -> str | None:
    vers = _versions.get(name)
    if not vers:
        return None
    return vers[-1]


def list_prompts() -> dict[str, list[str]]:
    return dict(_versions)


def prompt_names() -> list[str]:
    return list(_versions.keys())


def get_changelog(name: str, version: str) -> str:
    pt = get(name, version)
    if pt is None:
        return ""
    return pt.changelog


def render(pt: PromptTemplate, input_text: str) -> str:
    return pt.template.replace("{input_text}", input_text)


def count() -> int:
    return len(_cache)
