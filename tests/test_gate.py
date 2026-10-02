from evaluation.config import JSON_VALIDITY_MIN, JUDGE_MEAN_MIN


class TestGateThresholds:
    def test_json_validity_at_threshold(self):
        rate = 0.95
        assert rate >= JSON_VALIDITY_MIN

    def test_json_validity_below_threshold(self):
        rate = 0.949
        assert rate < JSON_VALIDITY_MIN

    def test_judge_mean_at_threshold(self):
        mean = 4.0
        assert mean >= JUDGE_MEAN_MIN

    def test_judge_mean_below_threshold(self):
        mean = 3.99
        assert mean < JUDGE_MEAN_MIN

    def test_threshold_values_match_spec(self):
        assert JSON_VALIDITY_MIN == 0.95
        assert JUDGE_MEAN_MIN == 4.0
