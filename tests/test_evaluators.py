import types

from evaluation.evaluators import json_schema_validity, summary_length_bounds


def _make_run(output):
    run = types.SimpleNamespace()
    run.outputs = {"output": output}
    return run


def _make_example(inputs=None, outputs=None):
    ex = types.SimpleNamespace()
    ex.inputs = inputs or {}
    ex.outputs = outputs or {}
    return ex


class TestJsonSchemaValidity:
    def test_valid_object(self):
        r = json_schema_validity(
            _make_run({"names": ["A"], "dates": [], "locations": []}),
            _make_example(),
        )
        assert r["score"] == 1

    def test_missing_key(self):
        r = json_schema_validity(
            _make_run({"names": ["A"], "dates": []}),
            _make_example(),
        )
        assert r["score"] == 0
        assert "locations" in r["comment"]

    def test_extra_key(self):
        r = json_schema_validity(
            _make_run({"names": [], "dates": [], "locations": [], "orgs": []}),
            _make_example(),
        )
        assert r["score"] == 0

    def test_string_json(self):
        r = json_schema_validity(
            _make_run('{"names": [], "dates": [], "locations": []}'),
            _make_example(),
        )
        assert r["score"] == 1

    def test_non_json_string(self):
        r = json_schema_validity(
            _make_run("just some text"),
            _make_example(),
        )
        assert r["score"] == 0

    def test_none_output(self):
        run = types.SimpleNamespace()
        run.outputs = {"output": None}
        r = json_schema_validity(run, _make_example())
        assert r["score"] == 0


class TestSummaryLengthBounds:
    def test_at_lower_bound(self):
        text = " ".join(["word"] * 20)
        r = summary_length_bounds(_make_run(text), _make_example())
        assert r["score"] == 1

    def test_below_lower_bound(self):
        text = " ".join(["word"] * 19)
        r = summary_length_bounds(_make_run(text), _make_example())
        assert r["score"] == 0
        assert "19" in r["comment"]

    def test_at_upper_bound(self):
        text = " ".join(["word"] * 60)
        r = summary_length_bounds(_make_run(text), _make_example())
        assert r["score"] == 1

    def test_above_upper_bound(self):
        text = " ".join(["word"] * 61)
        r = summary_length_bounds(_make_run(text), _make_example())
        assert r["score"] == 0
        assert "61" in r["comment"]

    def test_non_string_output(self):
        r = summary_length_bounds(_make_run({"key": "val"}), _make_example())
        assert r["score"] == 0
