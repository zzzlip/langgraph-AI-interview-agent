"""Worker 错误脱敏与失败映射辅助。"""

import re

_SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s]+"),
    re.compile(r"(?i)(token\s*[=:]\s*)[^\s]+"),
]
_STACK_PATTERN = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)
_ABSOLUTE_PATH_PATTERN = re.compile(r"(/Users|/private|/tmp|/var|[A-Za-z]:\\)[^\s\"']+")
_OBJECT_KEY_PATTERN = re.compile(r"(?i)(object_key\s*=\s*)[^\s]+")


def sanitize_feedback_text(text: str | None) -> str | None:
    """脱敏反馈文本，避免向平台事件暴露敏感细节。"""
    if text is None:
        return None

    sanitized = text
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
    sanitized = _STACK_PATTERN.sub("[STACK_REDACTED]", sanitized)
    sanitized = _ABSOLUTE_PATH_PATTERN.sub("[PATH_REDACTED]", sanitized)
    sanitized = _OBJECT_KEY_PATTERN.sub(r"\1[OBJECT_KEY_REDACTED]", sanitized)
    return sanitized
