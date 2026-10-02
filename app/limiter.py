import logging
import threading
import time

log = logging.getLogger(__name__)


class SlidingWindowLimiter:
    """Per-model sliding window rate limiter for requests and tokens."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._lock = threading.Lock()
        self._request_log: list[float] = []
        self._token_log: list[tuple[float, int]] = []

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_log and self._request_log[0] < cutoff:
            self._request_log.pop(0)
        while self._token_log and self._token_log[0][0] < cutoff:
            self._token_log.pop(0)

    def _current_tokens(self) -> int:
        return sum(t for _, t in self._token_log)

    def wait_for_capacity(self, estimated_tokens: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)

                if len(self._request_log) < self._rpm and (
                    self._current_tokens() + estimated_tokens <= self._tpm
                ):
                    self._request_log.append(now)
                    self._token_log.append((now, estimated_tokens))
                    return

                if self._request_log:
                    wait_req = 60.0 - (now - self._request_log[0])
                else:
                    wait_req = 0.0

                if self._token_log:
                    wait_tok = 60.0 - (now - self._token_log[0][0])
                else:
                    wait_tok = 0.0

                sleep_for = max(0.1, min(wait_req, wait_tok))

            log.debug("rate limiter sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)

    def reconcile_tokens(self, estimated: int, actual: int) -> None:
        diff = actual - estimated
        if diff <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._token_log.append((now, diff))


_limiters: dict[str, SlidingWindowLimiter] = {}
_global_lock = threading.Lock()


def get_limiter(model: str, rpm: int, tpm: int) -> SlidingWindowLimiter:
    with _global_lock:
        if model not in _limiters:
            _limiters[model] = SlidingWindowLimiter(rpm, tpm)
        return _limiters[model]
