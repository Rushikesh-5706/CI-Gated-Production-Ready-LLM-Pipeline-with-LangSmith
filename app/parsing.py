import json
import logging
import re

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def parse_summary(text: str) -> str:
    return text.strip()


def parse_entities(text: str) -> dict | str:
    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        log.warning("entity extraction returned non-object JSON (type=%s)", type(parsed).__name__)
        return stripped
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.match(stripped)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    log.warning("entity extraction output is not valid JSON, returning raw string")
    return stripped
