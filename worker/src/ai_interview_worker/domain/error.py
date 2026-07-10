from dataclasses import dataclass
from enum import Enum
from typing import Any


class WorkerFailureCode(str, Enum):
    """Worker 对平台暴露的稳定失败分类。"""

    INPUT_ERROR = "INPUT_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ARTIFACT_WRITE_ERROR = "ARTIFACT_WRITE_ERROR"


_DEFAULT_RETRYABLE: dict[WorkerFailureCode, bool] = {
    WorkerFailureCode.INPUT_ERROR: False,
    WorkerFailureCode.EXTERNAL_SERVICE_ERROR: True,
    WorkerFailureCode.INTERNAL_ERROR: True,
    WorkerFailureCode.ARTIFACT_WRITE_ERROR: True,
}


@dataclass
class WorkerFailure:
    """反馈给 Java 平台的脱敏失败语义。"""

    failure_code: WorkerFailureCode | str
    user_message: str
    failed_stage: str | None = None
    admin_summary: str | None = None
    retryable: bool | None = None

    def __post_init__(self) -> None:
        if isinstance(self.failure_code, str):
            try:
                parsed_failure_code = WorkerFailureCode(self.failure_code)
            except ValueError as exc:
                raise ValueError(f"unsupported failure_code: {self.failure_code}") from exc
            self.failure_code = parsed_failure_code
        if not self.user_message.strip():
            raise ValueError("user_message is required")
        if self.retryable is None:
            self.retryable = _DEFAULT_RETRYABLE[self.failure_code]

    def to_payload(self) -> dict[str, Any]:
        """生成失败事件 payload，禁止输出 internal_error_code。"""
        failure_code = self.failure_code
        if isinstance(failure_code, str):
            failure_code = WorkerFailureCode(failure_code)
        return {
            "status": "failed",
            "failed_stage": self.failed_stage,
            "failure_code": failure_code.value,
            "user_message": self.user_message,
            "admin_summary": self.admin_summary,
            "retryable": self.retryable,
        }


@dataclass
class TaskError:
    """兼容 spec01 dry-run 的旧错误类型。"""

    task_id: str
    error_code: str
    error_message: str
    retryable: bool = False
    stack_trace: str | None = None
