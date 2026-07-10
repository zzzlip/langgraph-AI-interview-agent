from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_interview_worker.domain.envelope import MessageEnvelope
from ai_interview_worker.domain.task_type import TaskType


@dataclass(frozen=True)
class InputFileRef:
    """平台传入的输入文件引用。"""

    file_id: str
    object_key: str
    file_type: str
    file_purpose: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputFileRef":
        """从 payload 字典构建输入文件引用。"""
        if not isinstance(data, dict):
            raise ValueError("input_files item must be an object")
        return cls(
            file_id=_require_str(data, "input_files.file_id", "file_id"),
            object_key=_require_str(data, "input_files.object_key", "object_key"),
            file_type=_require_str(data, "input_files.file_type", "file_type"),
            file_purpose=_require_str(data, "input_files.file_purpose", "file_purpose"),
        )


@dataclass
class TaskCommand:
    """平台下发给 Worker 的任务执行指令。"""

    task_id: str
    task_type: TaskType
    input_data: dict[str, Any] | None = None
    command_event_id: str = ""
    attempt: int = 1
    trace_id: str | None = None
    user_id: str = ""
    input_files: list[InputFileRef] | None = None
    input_summary: dict[str, Any] | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if isinstance(self.task_type, str):
            self.task_type = TaskType.from_value(self.task_type)
        if self.attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        if self.input_files is None:
            self.input_files = []
        if self.input_summary is None:
            self.input_summary = {"schema_version": 1}
        if self.input_data is None:
            self.input_data = {}

    @classmethod
    def from_envelope(cls, envelope: MessageEnvelope) -> "TaskCommand":
        """将合法 envelope 映射为应用层内部命令。"""
        payload = envelope.payload
        input_summary = payload.get("input_summary")
        if not isinstance(input_summary, dict) or "schema_version" not in input_summary:
            raise ValueError("input_summary.schema_version is required")

        raw_input_files = payload.get("input_files", [])
        if not isinstance(raw_input_files, list):
            raise ValueError("input_files must be a list")

        input_files = [InputFileRef.from_dict(item) for item in raw_input_files]

        return cls(
            command_event_id=envelope.event_id,
            task_id=envelope.task_id,
            task_type=envelope.task_type,
            attempt=envelope.attempt,
            trace_id=envelope.trace_id,
            user_id=str(payload.get("user_id", "")),
            input_files=input_files,
            input_summary=input_summary,
            input_data=payload,
        )


def _require_str(data: dict[str, Any], public_name: str, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{public_name} is required")
    return value
