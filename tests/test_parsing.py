from app.parsing import parse_entities, parse_summary


class TestParseSummary:
    def test_strips_whitespace(self):
        assert parse_summary("  hello world  ") == "hello world"

    def test_empty_string(self):
        assert parse_summary("   ") == ""


class TestParseEntities:
    def test_valid_json_object(self):
        result = parse_entities('{"names": ["Alice"], "dates": [], "locations": []}')
        assert isinstance(result, dict)
        assert result["names"] == ["Alice"]

    def test_fenced_json(self):
        text = '```json\n{"names": ["Bob"], "dates": [], "locations": []}\n```'
        result = parse_entities(text)
        assert isinstance(result, dict)
        assert result["names"] == ["Bob"]

    def test_prose_returns_string(self):
        text = "There are no entities in this text."
        result = parse_entities(text)
        assert isinstance(result, str)
        assert result == text

    def test_array_returns_string(self):
        text = '["not", "an", "object"]'
        result = parse_entities(text)
        assert isinstance(result, str)

    def test_invalid_json_returns_string(self):
        text = '{"broken": true'
        result = parse_entities(text)
        assert isinstance(result, str)
