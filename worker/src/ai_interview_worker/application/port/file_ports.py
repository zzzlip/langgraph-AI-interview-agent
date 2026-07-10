"""输入文件读取与产物写入 Port。"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ai_interview_worker.domain.artifact import OutputArtifact
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode


@dataclass(frozen=True)
class ReadInputResult:
    """输入文件读取结果。"""

    success: bool
    data: bytes | None = None
    mime_type: str = "application/octet-stream"
    failure: WorkerFailure | None = None


@dataclass(frozen=True)
class WriteArtifactResult:
    """产物写入结果。"""

    success: bool
    artifact: OutputArtifact | None = None
    failure: WorkerFailure | None = None


class InputFileReaderPort(Protocol):
    """按 object key 从 MinIO 读取输入文件。"""

    def get_object(self, object_key: str) -> ReadInputResult:
        """读取单个 object，返回结果或脱敏失败。"""
        ...


class ArtifactWriterPort(Protocol):
    """将产物写入 MinIO 并返回引用。"""

    def put_object(
        self,
        *,
        object_key: str,
        data: bytes,
        mime_type: str,
        display_name: str,
        clock: Callable[[], datetime],
    ) -> WriteArtifactResult:
        """写入单个产物并返回引用。"""
        ...


ARTIFACT_OBJECT_KEY_TEMPLATE = "tasks/{task_id}/attempt_{attempt}/{file_purpose}/{safe_file_name}"


def build_artifact_object_key(
    task_id: str,
    attempt: int,
    file_purpose: str,
    file_name: str,
) -> str:
    """生成产物 object key。"""
    safe_name = _sanitize_file_name(file_name)
    return ARTIFACT_OBJECT_KEY_TEMPLATE.format(
        task_id=task_id,
        attempt=attempt,
        file_purpose=file_purpose,
        safe_file_name=safe_name,
    )


def input_not_found_failure() -> WorkerFailure:
    return WorkerFailure(
        failure_code=WorkerFailureCode.INPUT_ERROR,
        user_message="输入文件不存在或无法访问",
        retryable=False,
    )


def artifact_write_failure() -> WorkerFailure:
    return WorkerFailure(
        failure_code=WorkerFailureCode.ARTIFACT_WRITE_ERROR,
        user_message="产物写入失败",
        retryable=True,
    )


def minio_connect_failure() -> WorkerFailure:
    return WorkerFailure(
        failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
        user_message="文件存储服务暂时不可用",
        retryable=True,
    )


def _sanitize_file_name(name: str) -> str:
    safe = []
    for char in name:
        if char in ("/", "\\", "\0") or ord(char) < 32:
            safe.append("_")
        else:
            safe.append(char)
    return "".join(safe)
