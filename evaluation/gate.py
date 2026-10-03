"""Quality gate logic as a pure function.

The gate examines the evaluation metrics produced by run_regression.py and
returns a GateDecision with a pass/fail result and the list of reasons for
any failure.
"""

from dataclasses import dataclass, field

from evaluation.config import JSON_VALIDITY_MIN, JUDGE_MEAN_MIN


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_gate(
    json_validity: float | None,
    judge_mean: float | None,
    scored: int,
    total: int,
) -> GateDecision:
    """Apply the quality gate to evaluation metrics.

    Parameters
    ----------
    json_validity:
        Mean json_schema_validity score across extraction examples (0.0-1.0).
        None means the evaluator produced no scores.
    judge_mean:
        Mean llm_as_judge_quality score across summarization examples (1.0-5.0).
        None means the evaluator produced no scores.
    scored:
        Number of examples for which every applicable evaluator returned a score.
    total:
        Total number of examples attempted.

    Returns
    -------
    GateDecision with ``passed=True`` only if all thresholds are met and
    scoring is complete.
    """
    reasons: list[str] = []

    if scored < total:
        reasons.append(
            f"{total - scored} of {total} examples unscored "
            f"(completeness check failed)"
        )

    if json_validity is None:
        reasons.append("json_schema_validity produced no scores")
    elif json_validity < JSON_VALIDITY_MIN:
        reasons.append(
            f"json_schema_validity {json_validity:.4f} < threshold {JSON_VALIDITY_MIN}"
        )

    if judge_mean is None:
        reasons.append("llm_as_judge_quality produced no scores")
    elif judge_mean < JUDGE_MEAN_MIN:
        reasons.append(
            f"llm_as_judge_quality mean {judge_mean:.4f} < threshold {JUDGE_MEAN_MIN}"
        )

    return GateDecision(passed=len(reasons) == 0, reasons=reasons)
