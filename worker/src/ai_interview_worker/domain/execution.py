"""Worker 执行状态模型。"""

from enum import Enum


class WorkerExecutionStatus(str, Enum):
    """一次任务命令在 Worker 内部的执行状态。"""

    RECEIVED = "received"
    DEDUPLICATED = "deduplicated"
    PROCESSING = "processing"
    ARTIFACT_WRITING = "artifact_writing"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES = {
    WorkerExecutionStatus.DEDUPLICATED,
    WorkerExecutionStatus.COMPLETED,
    WorkerExecutionStatus.FAILED,
}


def ensure_transition_allowed(current: WorkerExecutionStatus, target: WorkerExecutionStatus) -> None:
    """校验 Worker 执行状态流转。"""
    if current in TERMINAL_STATUSES and target is WorkerExecutionStatus.PROCESSING:
        raise ValueError(f"cannot transition from {current.value} to {target.value}")
