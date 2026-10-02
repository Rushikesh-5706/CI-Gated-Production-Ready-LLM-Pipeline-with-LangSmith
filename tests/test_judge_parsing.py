import json
import types
from unittest.mock import patch


def _mock_chat_completion(text: str):
    """Build a mock return value matching app.llm.chat_completion's output shape."""
    result = types.SimpleNamespace(
        text=text,
        model="test",
        prompt_tokens=10,
        completion_tokens=10,
        latency_seconds=0.1,
    )
    return {
        "result": result,
        "usage_metadata": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        "ls_provider": "groq",
        "ls_model_name": "test",
        "ls_temperature": 0,
    }


class TestJudgeParsing:
    def test_valid_response(self):
        resp = json.dumps(
            {"relevance_score": 4, "conciseness_score": 5, "justification": "Good summary."}
        )
        with patch("evaluation.judge.chat_completion", return_value=_mock_chat_completion(resp)):
            from evaluation.judge import llm_as_judge_quality

            run = types.SimpleNamespace(outputs={"output": "A summary."})
            example = types.SimpleNamespace(inputs={"input_text": "Original text here."})
            r = llm_as_judge_quality(run, example)
            assert r["score"] == 4.5
            assert "relevance=4" in r["comment"]

    def test_out_of_range_scores(self):
        resp = json.dumps({"relevance_score": 6, "conciseness_score": 0, "justification": "Bad."})
        with patch("evaluation.judge.chat_completion", return_value=_mock_chat_completion(resp)):
            from evaluation.judge import llm_as_judge_quality

            run = types.SimpleNamespace(outputs={"output": "A summary."})
            example = types.SimpleNamespace(inputs={"input_text": "Original text."})
            r = llm_as_judge_quality(run, example)
            assert r["score"] == 0

    def test_malformed_json(self):
        with patch(
            "evaluation.judge.chat_completion", return_value=_mock_chat_completion("not json")
        ):
            from evaluation.judge import llm_as_judge_quality

            run = types.SimpleNamespace(outputs={"output": "A summary."})
            example = types.SimpleNamespace(inputs={"input_text": "Original text."})
            r = llm_as_judge_quality(run, example)
            assert r["score"] == 0
            assert "failed" in r["comment"]
