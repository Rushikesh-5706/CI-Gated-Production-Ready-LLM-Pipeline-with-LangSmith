"""Tests for the quality gate pure function."""


from evaluation.config import JSON_VALIDITY_MIN, JUDGE_MEAN_MIN
from evaluation.gate import evaluate_gate


class TestThresholdValues:
    def test_json_validity_threshold(self):
        assert JSON_VALIDITY_MIN == 0.95

    def test_judge_mean_threshold(self):
        assert JUDGE_MEAN_MIN == 4.0


class TestEvaluateGate:
    def test_all_pass(self):
        d = evaluate_gate(json_validity=1.0, judge_mean=5.0, scored=30, total=30)
        assert d.passed is True
        assert d.reasons == []

    def test_json_validity_at_exact_threshold_passes(self):
        d = evaluate_gate(json_validity=0.95, judge_mean=4.0, scored=30, total=30)
        assert d.passed is True

    def test_json_validity_29_of_30_passes(self):
        # 29/30 = 0.9667 > 0.95
        d = evaluate_gate(json_validity=29 / 30, judge_mean=4.0, scored=30, total=30)
        assert d.passed is True

    def test_json_validity_28_of_30_fails(self):
        # 28/30 = 0.9333 < 0.95
        d = evaluate_gate(json_validity=28 / 30, judge_mean=4.0, scored=30, total=30)
        assert d.passed is False
        assert any("json_schema_validity" in r for r in d.reasons)

    def test_judge_mean_at_exact_threshold_passes(self):
        d = evaluate_gate(json_validity=1.0, judge_mean=4.0, scored=30, total=30)
        assert d.passed is True

    def test_judge_mean_just_below_threshold_fails(self):
        d = evaluate_gate(json_validity=1.0, judge_mean=3.99, scored=30, total=30)
        assert d.passed is False
        assert any("llm_as_judge_quality" in r for r in d.reasons)

    def test_incomplete_scoring_fails(self):
        d = evaluate_gate(json_validity=1.0, judge_mean=5.0, scored=25, total=30)
        assert d.passed is False
        assert any("unscored" in r for r in d.reasons)

    def test_none_json_validity_fails(self):
        d = evaluate_gate(json_validity=None, judge_mean=5.0, scored=30, total=30)
        assert d.passed is False
        assert any("json_schema_validity" in r for r in d.reasons)

    def test_none_judge_mean_fails(self):
        d = evaluate_gate(json_validity=1.0, judge_mean=None, scored=30, total=30)
        assert d.passed is False
        assert any("llm_as_judge_quality" in r for r in d.reasons)

    def test_multiple_failures_all_reported(self):
        d = evaluate_gate(json_validity=0.8, judge_mean=3.0, scored=20, total=30)
        assert d.passed is False
        assert len(d.reasons) == 3

    def test_reasons_contain_actual_values(self):
        d = evaluate_gate(json_validity=0.80, judge_mean=3.5, scored=30, total=30)
        assert "0.8000" in " ".join(d.reasons)
        assert "3.5000" in " ".join(d.reasons)

    def test_reasons_contain_thresholds(self):
        d = evaluate_gate(json_validity=0.90, judge_mean=3.5, scored=30, total=30)
        assert "0.95" in " ".join(d.reasons)
        assert "4.0" in " ".join(d.reasons)
