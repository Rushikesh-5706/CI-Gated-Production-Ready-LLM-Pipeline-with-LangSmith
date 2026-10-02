import json
import logging
from pathlib import Path

import jsonschema
from langsmith.schemas import Example, Run

from evaluation.config import (
    EVALUATOR_JSON_SCHEMA,
    EVALUATOR_LENGTH,
    WORD_COUNT_MAX,
    WORD_COUNT_MIN,
)

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "entities.schema.json"
_ENTITY_SCHEMA: dict | None = None


def _load_schema() -> dict:
    global _ENTITY_SCHEMA
    if _ENTITY_SCHEMA is None:
        _ENTITY_SCHEMA = json.loads(_SCHEMA_PATH.read_text())
    return _ENTITY_SCHEMA


def json_schema_validity(run: Run, example: Example) -> dict:
    output = run.outputs.get("output") if run.outputs else None

    if output is None:
        return {"key": EVALUATOR_JSON_SCHEMA, "score": 0, "comment": "no output produced"}

    obj = output
    if isinstance(output, str):
        try:
            obj = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return {
                "key": EVALUATOR_JSON_SCHEMA,
                "score": 0,
                "comment": f"output is not valid JSON: {output[:200]}",
            }

    if not isinstance(obj, dict):
        return {
            "key": EVALUATOR_JSON_SCHEMA,
            "score": 0,
            "comment": f"output is {type(obj).__name__}, not an object",
        }

    schema = _load_schema()
    try:
        jsonschema.validate(obj, schema)
        return {"key": EVALUATOR_JSON_SCHEMA, "score": 1, "comment": "valid"}
    except jsonschema.ValidationError as exc:
        return {
            "key": EVALUATOR_JSON_SCHEMA,
            "score": 0,
            "comment": f"schema violation: {exc.message}",
        }


def summary_length_bounds(run: Run, example: Example) -> dict:
    output = run.outputs.get("output") if run.outputs else None

    if output is None:
        return {"key": EVALUATOR_LENGTH, "score": 0, "comment": "no output produced"}

    if not isinstance(output, str):
        return {
            "key": EVALUATOR_LENGTH,
            "score": 0,
            "comment": f"output is {type(output).__name__}, not a string",
        }

    count = len(output.split())
    if WORD_COUNT_MIN <= count <= WORD_COUNT_MAX:
        return {"key": EVALUATOR_LENGTH, "score": 1, "comment": f"word count {count}"}

    return {
        "key": EVALUATOR_LENGTH,
        "score": 0,
        "comment": f"word count {count} outside [{WORD_COUNT_MIN}, {WORD_COUNT_MAX}]",
    }
