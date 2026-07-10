"""Worker 本地命令幂等记录。"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ProcessedCommandStatus(str, Enum):
    """本地幂等记录状态。"""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProcessedCommand:
    """Worker 单实例本地幂等记录，不作为平台业务事实。"""

    command_event_id: str
    task_id: str
    attempt: int
    task_type: str
    status: ProcessedCommandStatus
    payload_hash: str
    trace_id: str
    created_at: datetime
    updated_at: datetime
    action: str = ""

    def __post_init__(self) -> None:
        if self.attempt <= 0:
            raise ValueError("attempt must be a positive integer")
