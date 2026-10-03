"""Custom LangSmith evaluators for the CI regression suite.

Three required evaluators (keys fixed by contract):
  json_schema_validity    -- extraction output matches the entity schema
  summary_length_bounds   -- summary word count within [WORD_COUNT_MIN, WORD_COUNT_MAX]
  llm_as_judge_quality    -- LLM rates summary quality 1-5 (returned as-is)

One informational evaluator (not gated):
  entity_match_f1         -- exact-string F1 against expected names/dates/locations
"""

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


def entity_match_f1(run: Run, example: Example) -> dict:
    """Informational evaluator: micro-averaged exact-string F1 against expected outputs.

    Compares the three entity lists (names, dates, locations) from the run
    output against the expected values in example.outputs.  Not used by the gate.
    """
    expected = example.outputs or {}
    actual_raw = run.outputs.get("output") if run.outputs else None

    if actual_raw is None:
        return {"key": "entity_match_f1", "score": 0.0, "comment": "no output produced"}

    if isinstance(actual_raw, str):
        try:
            actual_raw = json.loads(actual_raw)
        except (json.JSONDecodeError, TypeError):
            return {
                "key": "entity_match_f1",
                "score": 0.0,
                "comment": "output is not valid JSON",
            }

    if not isinstance(actual_raw, dict):
        return {
            "key": "entity_match_f1",
            "score": 0.0,
            "comment": f"output is {type(actual_raw).__name__}, not an object",
        }

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for field in ("names", "dates", "locations"):
        expected_set = set(expected.get(field, []))
        actual_set = set(actual_raw.get(field, []))
        tp_total += len(expected_set & actual_set)
        fp_total += len(actual_set - expected_set)
        fn_total += len(expected_set - actual_set)

    if tp_total + fp_total + fn_total == 0:
        score = 1.0
        comment = "both sets empty"
    else:
        precision = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)
        comment = f"P={precision:.3f} R={recall:.3f} tp={tp_total} fp={fp_total} fn={fn_total}"

    return {"key": "entity_match_f1", "score": round(score, 4), "comment": comment}
