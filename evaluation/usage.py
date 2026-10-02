import threading


class TokenAccumulator:
    """Thread-safe accumulator for token usage across evaluation runs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._usage: dict[str, dict[str, int]] = {}

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            if model not in self._usage:
                self._usage[model] = {"prompt_tokens": 0, "completion_tokens": 0}
            self._usage[model]["prompt_tokens"] += prompt_tokens
            self._usage[model]["completion_tokens"] += completion_tokens

    def totals(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                model: {
                    "prompt_tokens": counts["prompt_tokens"],
                    "completion_tokens": counts["completion_tokens"],
                    "total_tokens": counts["prompt_tokens"] + counts["completion_tokens"],
                }
                for model, counts in self._usage.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._usage.clear()


_active: TokenAccumulator | None = None


def get_accumulator() -> TokenAccumulator | None:
    return _active


def start_accumulator() -> TokenAccumulator:
    global _active
    _active = TokenAccumulator()
    return _active


def stop_accumulator() -> TokenAccumulator | None:
    global _active
    acc = _active
    _active = None
    return acc
