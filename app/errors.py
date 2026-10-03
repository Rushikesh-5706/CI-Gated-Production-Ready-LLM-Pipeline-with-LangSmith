class PipelineError(Exception):
    pass


class PromptNotFoundError(PipelineError):
    def __init__(self, message: str, available: list[str] | None = None):
        super().__init__(message)
        self.available = available or []


class VersionNotFoundError(PipelineError):
    def __init__(self, message: str, available: list[str] | None = None):
        super().__init__(message)
        self.available = available or []


class LLMNotConfiguredError(PipelineError):
    pass


class LLMRateLimitError(PipelineError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(PipelineError):
    pass


class LLMUpstreamError(PipelineError):
    pass


class LLMEmptyResponseError(PipelineError):
    pass
