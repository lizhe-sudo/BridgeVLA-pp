"""Typed Astra evaluator errors and fail-closed diagnostic sanitization."""

import re


MAX_DIAGNOSTIC_CHARS = 4096
MAX_REJECTED_OUTPUT_BYTES = 16384


class AstraEvaluationError(RuntimeError):
    error_code = "astra_evaluation_error"

    def __init__(self, message="", error_code=None):
        super().__init__(message)
        if error_code:
            self.error_code = str(error_code)


class InvalidPolicyOutput(AstraEvaluationError):
    error_code = "invalid_policy_output"


class PolicyToolViolation(AstraEvaluationError):
    error_code = "policy_tool_violation"


class InferenceDeadlineExceeded(AstraEvaluationError):
    error_code = "inference_deadline_exceeded"


class ModelServiceError(AstraEvaluationError):
    error_code = "model_service_error"


class SimulatorInfrastructureError(AstraEvaluationError):
    error_code = "simulator_infrastructure_error"


class CoreArtifactWriteError(AstraEvaluationError):
    error_code = "core_artifact_write_error"


class UnknownEvaluationError(AstraEvaluationError):
    error_code = "unclassified_error"


_SECRET_ASSIGNMENT_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*"),
    re.compile(
        r"(?i)([\"']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
        r"auth(?:orization)?|password|secret|credential)[\"']?\s*[:=]\s*)"
        r"([\"']?)([^\s,;\"'{}\]]+)"
    ),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,})\b"),
)


def sanitize_diagnostic(value, limit=MAX_DIAGNOSTIC_CHARS):
    """Return bounded, redacted text; never fall back to raw text on failure."""
    raw_length = None
    try:
        text = str(value)
        raw_length = len(text)
        redacted = text
        for index, pattern in enumerate(_SECRET_ASSIGNMENT_PATTERNS):
            if index == 0:
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
            elif index == 1:
                redacted = pattern.sub(r"\1\2[REDACTED]", redacted)
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
        redacted = "".join(
            char if char in "\n\r\t" or ord(char) >= 32 else " "
            for char in redacted
        )
        truncated = len(redacted) > limit
        return {
            "text": redacted[:limit],
            "original_chars": raw_length,
            "truncated": truncated,
            "redaction_failed": False,
        }
    except Exception:
        return {
            "text": "<diagnostic redaction failed; content omitted>",
            "original_chars": raw_length,
            "truncated": raw_length is None or raw_length > limit,
            "redaction_failed": True,
        }


def safe_exception_record(exc):
    """Serialize an exception without exposing unfiltered exception text."""
    try:
        detail = sanitize_diagnostic(exc)
    except Exception:
        detail = {
            "text": "<diagnostic redaction failed; content omitted>",
            "original_chars": None,
            "truncated": True,
            "redaction_failed": True,
        }
    return {
        "error_type": type(exc).__name__,
        "error_code": getattr(exc, "error_code", "unclassified_error"),
        "error_summary": detail["text"],
        "error_summary_truncated": detail["truncated"],
        "error_summary_redaction_failed": detail["redaction_failed"],
    }
