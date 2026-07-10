"""RabbitMQ 消息 envelope 领域模型。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_interview_worker.domain.task_type import COMMAND_EVENT_TYPES, TaskType


@dataclass(frozen=True)
class MessageEnvelope:
    """平台与 Worker 之间传递的统一消息包裹结构。"""

    schema_version: int
    event_id: str
    event_type: str
    task_id: str
    task_type: TaskType
    attempt: int
    trace_id: str
    occurred_at: datetime | None
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageEnvelope":
        """从字典构建 envelope，并执行基础契约校验。"""
        if not isinstance(data, dict):
            raise ValueError("envelope must be an object")

        required_fields = (
            "schema_version",
            "event_id",
            "event_type",
            "task_id",
            "task_type",
            "attempt",
            "trace_id",
            "payload",
        )
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"missing {field_name}")

        if data["schema_version"] != 1:
            raise ValueError("schema_version must be 1")

        task_type = TaskType.from_value(str(data["task_type"]))
        expected_event_type = COMMAND_EVENT_TYPES[task_type]
        event_type = data["event_type"]
        if event_type not in COMMAND_EVENT_TYPES.values():
            raise ValueError(f"unsupported event_type: {event_type}")
        if event_type != expected_event_type:
            raise ValueError(f"event_type does not match task_type: {event_type} != {task_type.value}")

        attempt = data["attempt"]
        if not isinstance(attempt, int) or attempt <= 0:
            raise ValueError("attempt must be a positive integer")

        payload = data["payload"]
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")

        return cls(
            schema_version=1,
            event_id=_require_non_empty_str(data, "event_id"),
            event_type=_require_non_empty_str(data, "event_type"),
            task_id=_require_non_empty_str(data, "task_id"),
            task_type=task_type,
            attempt=attempt,
            trace_id=_require_non_empty_str(data, "trace_id"),
            occurred_at=_parse_occurred_at(data.get("occurred_at")),
            payload=payload,
        )


def _require_non_empty_str(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _parse_occurred_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("occurred_at must be an ISO-8601 string")
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("occurred_at must be an ISO-8601 string") from exc
