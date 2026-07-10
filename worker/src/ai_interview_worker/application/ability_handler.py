"""AbilityHandler 抽象接口与 AbilityRouter。"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ai_interview_worker.application.port.ai_gateway_port import AiGatewayPort
from ai_interview_worker.application.port.file_ports import ArtifactWriterPort, ReadInputResult
from ai_interview_worker.domain.command import TaskCommand
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.result import WorkerSuccessResult
from ai_interview_worker.domain.task_type import TaskType


class ProgressReporter(Protocol):
    """进度上报回调。"""

    def __call__(self, *, stage: str, progress: int, message: str) -> None:
        ...


class InputFilePort(Protocol):
    """输入文件读取 Port。"""

    def get_object(self, object_key: str) -> ReadInputResult:
        ...


@dataclass
class AbilityContext:
    """Handler 执行上下文。"""

    command: TaskCommand
    progress_reporter: ProgressReporter
    minio_reader: InputFilePort | None = None
    artifact_writer: ArtifactWriterPort | None = None
    ai_gateway: AiGatewayPort | None = None
    clock: Callable[[], datetime] | None = None
    event_id_supplier: Callable[[], str] | None = None
    feedback_publisher: Callable[[dict[str, Any]], None] | None = None


class AbilityHandler(Protocol):
    """AI 能力单元统一接口。"""

    def handle(self, ctx: AbilityContext) -> WorkerSuccessResult:
        ...


@dataclass
class RouterResult:
    """路由结果。"""

    success: bool
    handler: AbilityHandler | None = None
    failure: WorkerFailure | None = None


class AbilityRouter:
    """根据 task_type 路由到对应 Handler。"""

    def __init__(self, handlers: dict[TaskType, AbilityHandler]) -> None:
        self._handlers = handlers

    def route(self, task_type: TaskType) -> RouterResult:
        handler = self._handlers.get(task_type)
        if handler is None:
            return RouterResult(
                success=False,
                failure=WorkerFailure(
                    failure_code=WorkerFailureCode.INPUT_ERROR,
                    user_message=f"不支持的任务类型: {task_type.value}",
                    failed_stage="task_routing",
                    retryable=False,
                ),
            )
        return RouterResult(success=True, handler=handler)
