from __future__ import annotations
from typing import Any
class AppError(Exception):
    code = "app_error"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self, request_id: str) -> dict[str, Any]:
        """Serialize for the HTTP response envelope."""
        body: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
            "request_id": request_id,
        }
        if self.details is not None and self.details:
            body["details"] = self.details
        return body

# ------ Client side errors: 4xx --------------------------------
class BadRequestError(AppError):
    """400: malformed request from the client."""
    code = "bad_request"
    status_code = 400


class UnauthorizedError(AppError):
    """401: authentication required or failed."""
    code = "unauthorized"
    status_code = 401


class ForbiddenError(AppError):
    """403: authenticated but not permitted."""
    code = "forbidden"
    status_code = 403


class NotFoundError(AppError):
    """404: resource does not exist."""
    code = "not_found"
    status_code = 404


class PayloadTooLargeError(AppError):
    """413: request body exceeds configured size limit."""
    code = "payload_too_large"
    status_code = 413


class UnsupportedFileTypeError(AppError):
    """415: file uploaded but its type isn't handled."""
    code = "unsupported_file_type"
    status_code = 415


class UnprocessableEntityError(AppError):
    """422: request is well-formed but semantically invalid."""
    code = "unprocessable_entity"
    status_code = 422


class RateLimitError(AppError):
    """429: too many requests."""
    code = "rate_limited"
    status_code = 429


# --------- Domain Specific Errors -----------------------------------

class PromptInjectionError(BadRequestError):
    """
    A user query looks like a prompt-injection attempt. Not a generic bad
    request — it deserves its own code so metrics can count these.
    """
    code = "prompt_injection_detected"


class LowQualityDocumentError(UnprocessableEntityError):
    """
    Ingested document extracted successfully but is too short or noisy to
    be useful. Distinct from a generic ingest failure.
    """
    code = "low_quality_document"


class IngestionError(UnprocessableEntityError):
    """Generic ingest-time failure (parser error, malformed file, etc.)."""
    code = "ingestion_failed"


# ------- 5xx Errors -------------------------------------
class InternalError(AppError):
    """500: unexpected server error. Prefer specific subclasses."""
    code = "internal_error"
    status_code = 500


class ExternalDependencyError(AppError):
    """503: an upstream we depend on failed or is unavailable."""
    code = "external_dependency_failed"
    status_code = 503


class GenerationTimeoutError(AppError):
    """
    504: the generator (SAP or llama.cpp) didn't return within its budget.
    Distinct from a generic 503 so alerting can page on timeouts specifically.
    """
    code = "generation_timeout"
    status_code = 504