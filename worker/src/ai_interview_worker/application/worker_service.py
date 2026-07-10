"""Worker 命令处理应用服务。"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from ai_interview_worker.application.ability_handler import AbilityContext, AbilityRouter
from ai_interview_worker.application.feedback_service import FeedbackPublishError, FeedbackService
from ai_interview_worker.application.idempotency import IdempotencyDecision
from ai_interview_worker.application.port.ai_gateway_port import AiGatewayPort
from ai_interview_worker.application.port.file_ports import ArtifactWriterPort, InputFileReaderPort
from ai_interview_worker.domain.command import TaskCommand
from ai_interview_worker.domain.envelope import MessageEnvelope
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.task_type import TaskType


class IdempotencyStorePort(Protocol):
    """命令幂等存储 Port。"""

    def try_start(
        self,
        *,
        command_event_id: str,
        task_id: str,
        attempt: int,
        action: str,
        task_type: str,
        payload_hash: str,
        trace_id: str,
        now: datetime,
        timeout_seconds: int,
    ) -> IdempotencyDecision:
        """尝试建立命令处理栅栏。"""

    def mark_completed(self, command_event_id: str, now: datetime) -> None:
        """终态结果确认发布后标记完成。"""

    def mark_failed(self, command_event_id: str, now: datetime) -> None:
        """终态失败确认发布后标记失败。"""


class FeedbackPublisherPort(Protocol):
    """Worker 反馈发布 Port。"""

    def publish(self, event: dict[str, Any]) -> None:
        """发布反馈事件。"""


class HandleDecisionType(str, Enum):
    """命令处理入口的 ACK/NACK 决策。"""

    PROCESS = "process"
    ACK = "ack"
    NACK_REQUEUE = "nack_requeue"


@dataclass(frozen=True)
class HandleResult:
    """命令处理入口结果。"""

    decision_type: HandleDecisionType
    entered_ai_execution: bool
    reason: str


class WorkerCommandService:
    """编排命令契约校验、幂等栅栏和 consumed 反馈。"""

    def __init__(
        self,
        *,
        idempotency_store: IdempotencyStorePort,
        feedback_publisher: FeedbackPublisherPort,
        worker_id: str,
        event_id_supplier: Callable[[], str],
        clock: Callable[[], datetime],
        timeout_seconds: int,
        ability_router: AbilityRouter | None = None,
        minio_reader: InputFileReaderPort | None = None,
        artifact_writer: ArtifactWriterPort | None = None,
        ai_gateway: AiGatewayPort | None = None,
    ) -> None:
        self.idempotency_store = idempotency_store
        self.feedback_publisher = feedback_publisher
        self.worker_id = worker_id
        self.feedback_service = FeedbackService(event_id_supplier, clock)
        self.clock = clock
        self.timeout_seconds = timeout_seconds
        self.ability_router = ability_router
        self.minio_reader = minio_reader
        self.artifact_writer = artifact_writer
        self.ai_gateway = ai_gateway

    def handle(self, raw_message: dict[str, Any] | str | bytes) -> HandleResult:
        """处理单条命令消息；未配置能力路由时只执行 consumed 前置闭环。"""
        command = _parse_command_message(raw_message)
        decision = self._try_start(command)

        if decision.should_ack:
            return HandleResult(HandleDecisionType.ACK, entered_ai_execution=False, reason=decision.reason)
        if decision.should_nack_requeue:
            return HandleResult(HandleDecisionType.NACK_REQUEUE, entered_ai_execution=False, reason=decision.reason)

        try:
            self.feedback_publisher.publish(self.feedback_service.consumed_event(command, self.worker_id))
        except FeedbackPublishError:
            return HandleResult(
                HandleDecisionType.NACK_REQUEUE,
                entered_ai_execution=False,
                reason="consumed_publish_failed",
            )

        if self.ability_router is not None:
            return self._execute_command(command)

        return HandleResult(HandleDecisionType.PROCESS, entered_ai_execution=False, reason="consumed_published")

    def _try_start(self, command: TaskCommand) -> IdempotencyDecision:
        return self.idempotency_store.try_start(
            command_event_id=command.command_event_id,
            task_id=command.task_id,
            attempt=command.attempt,
            action=_command_action_scope(command),
            task_type=command.task_type.value,
            payload_hash=_payload_hash(command.input_data),
            trace_id=command.trace_id or "",
            now=self.clock(),
            timeout_seconds=self.timeout_seconds,
        )

    def _execute_command(self, command: TaskCommand) -> HandleResult:
        route_result = self.ability_router.route(command.task_type) if self.ability_router else None
        if route_result is None:
            return HandleResult(HandleDecisionType.PROCESS, entered_ai_execution=False, reason="router_not_configured")
        if not route_result.success or route_result.handler is None:
            failure = route_result.failure or WorkerFailure(
                WorkerFailureCode.INPUT_ERROR,
                f"不支持的任务类型: {command.task_type.value}",
                failed_stage="task_routing",
                retryable=False,
            )
            return self._publish_failed(command, failure, reason="task_routing_failed")

        def report_progress(*, stage: str, progress: int, message: str) -> None:
            self.feedback_service.publish_progress(
                self.feedback_publisher,
                command,
                status="running",
                stage=stage,
                progress=progress,
                message=message,
            )

        try:
            ctx = AbilityContext(
                command=command,
                progress_reporter=report_progress,
                minio_reader=self.minio_reader,
                artifact_writer=self.artifact_writer,
                ai_gateway=self.ai_gateway,
                clock=self.clock,
                event_id_supplier=self.feedback_service.event_id_supplier,
                feedback_publisher=self.feedback_publisher.publish,
            )
            result = route_result.handler.handle(ctx)
            if _is_intermediate_only_result(command, result.result_summary):
                self.idempotency_store.mark_completed(command.command_event_id, self.clock())
                return HandleResult(
                    HandleDecisionType.ACK,
                    entered_ai_execution=True,
                    reason="intermediate_feedback_published",
                )
            self.feedback_publisher.publish(self.feedback_service.result_event(command, result))
            self.idempotency_store.mark_completed(command.command_event_id, self.clock())
            return HandleResult(HandleDecisionType.ACK, entered_ai_execution=True, reason="result_published")
        except FeedbackPublishError:
            return HandleResult(
                HandleDecisionType.NACK_REQUEUE,
                entered_ai_execution=True,
                reason="terminal_feedback_publish_failed",
            )
        except Exception as exc:
            failure = _failure_from_exception(exc)
            return self._publish_failed(command, failure, reason="handler_failed")

    def _publish_failed(self, command: TaskCommand, failure: WorkerFailure, *, reason: str) -> HandleResult:
        try:
            self.feedback_publisher.publish(self.feedback_service.failed_event(command, failure))
            self.idempotency_store.mark_failed(command.command_event_id, self.clock())
            return HandleResult(HandleDecisionType.ACK, entered_ai_execution=True, reason=reason)
        except FeedbackPublishError:
            return HandleResult(
                HandleDecisionType.NACK_REQUEUE,
                entered_ai_execution=True,
                reason="failed_feedback_publish_failed",
            )


def _payload_hash(payload: dict[str, Any] | None) -> str:
    normalized = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _command_action_scope(command: TaskCommand) -> str:
    action = _command_action(command)
    if command.task_type == TaskType.MOCK_INTERVIEW and action == "process_answer":
        summary = command.input_summary if isinstance(command.input_summary, dict) else {}
        raw_round_id = summary.get("current_round_id")
        round_id = raw_round_id.strip() if isinstance(raw_round_id, str) and raw_round_id.strip() else ""
        return f"{action}:{round_id}" if round_id else action
    return action


def _command_action(command: TaskCommand) -> str:
    summary = command.input_summary if isinstance(command.input_summary, dict) else {}
    raw_action = summary.get("action")
    action = raw_action.strip() if isinstance(raw_action, str) and raw_action.strip() else command.task_type.value
    return action


def _parse_command_message(message: dict[str, Any] | str | bytes) -> TaskCommand:
    if isinstance(message, bytes):
        data = json.loads(message.decode("utf-8"))
    elif isinstance(message, str):
        data = json.loads(message)
    else:
        data = message
    return TaskCommand.from_envelope(MessageEnvelope.from_dict(data))


def _failure_from_exception(exc: Exception) -> WorkerFailure:
    if exc.args and isinstance(exc.args[0], WorkerFailure):
        return exc.args[0]
    return WorkerFailure(
        WorkerFailureCode.INTERNAL_ERROR,
        "Worker 执行过程中发生内部错误",
        failed_stage="worker_execution",
        admin_summary=str(exc)[:200],
        retryable=True,
    )


def _is_intermediate_only_result(command: TaskCommand, result_summary: dict[str, Any]) -> bool:
    if command.task_type == TaskType.ALGORITHM_TEST:
        return result_summary.get("phase") == "problem_set"
    if command.task_type == TaskType.MOCK_INTERVIEW:
        return _command_action(command) in {
            "generate_interview_questions",
            "process_answer",
        }
    return False
