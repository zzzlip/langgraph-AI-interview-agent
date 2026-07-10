"""RabbitMQ 命令消息映射。"""

import json
from typing import Any

from ai_interview_worker.domain.command import TaskCommand
from ai_interview_worker.domain.envelope import MessageEnvelope


def parse_command_message(message: bytes | str | dict[str, Any]) -> TaskCommand:
    """解析 spec05 任务命令消息。"""
    if isinstance(message, bytes):
        raw_message = message.decode("utf-8")
        data = json.loads(raw_message)
    elif isinstance(message, str):
        data = json.loads(message)
    else:
        data = message

    envelope = MessageEnvelope.from_dict(data)
    return TaskCommand.from_envelope(envelope)
