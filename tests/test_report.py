from evaluation.report import MARKER, build_report


class TestReportRendering:
    def test_marker_present(self):
        results = {
            "gate_pass": True,
            "failed_reasons": [],
            "metrics": [
                {
                    "prompt": "summarize_text",
                    "version": "v2",
                    "metric": "llm_as_judge_quality",
                    "value": 4.5,
                    "threshold": 4.0,
                    "status": "PASS",
                },
            ],
            "datasets": [{"name": "summarization-regression", "expected": 30, "scored": 30}],
        }
        report = build_report(results, None, {"total": 0.001}, {}, "abc1234", {}, [])
        assert MARKER in report
        assert "PASSED" in report

    def test_failed_report(self):
        results = {
            "gate_pass": False,
            "failed_reasons": ["judge mean below threshold"],
            "metrics": [
                {
                    "prompt": "summarize_text",
                    "version": "v1",
                    "metric": "llm_as_judge_quality",
                    "value": 3.5,
                    "threshold": 4.0,
                    "status": "FAIL",
                },
            ],
            "datasets": [],
        }
        report = build_report(results, None, {"total": 0.0}, {}, "def5678", {}, [])
        assert "FAILED" in report
        assert "judge mean below threshold" in report

    def test_delta_computation(self):
        results = {
            "gate_pass": True,
            "failed_reasons": [],
            "metrics": [
                {
                    "prompt": "summarize_text",
                    "version": "v2",
                    "metric": "llm_as_judge_quality",
                    "value": 4.5,
                    "threshold": 4.0,
                    "status": "PASS",
                },
            ],
            "datasets": [],
        }
        baseline = {"summarize_text_llm_as_judge_quality": 4.0}
        report = build_report(results, baseline, {"total": 0.0}, {}, "abc", {}, [])
        assert "+0.500" in report
