"""Worker 反馈事件生成服务。"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from ai_interview_worker.application.error_mapper import sanitize_feedback_text
from ai_interview_worker.domain.command import TaskCommand
from ai_interview_worker.domain.error import WorkerFailure
from ai_interview_worker.domain.result import WorkerSuccessResult

ALLOWED_PROGRESS_STATUSES = {"queued", "running", "waiting_user_input"}


class FeedbackPublishError(RuntimeError):
    """反馈事件发布失败。"""


class FeedbackService:
    """生成符合 spec05 契约的 Worker 反馈事件。"""

    def __init__(self, event_id_supplier: Callable[[], str], clock: Callable[[], datetime]) -> None:
        self.event_id_supplier = event_id_supplier
        self.clock = clock

    def consumed_event(self, command: TaskCommand, worker_id: str) -> dict[str, Any]:
        """生成 task.command.consumed 反馈事件。"""
        event = self._base_event(command, "task.command.consumed")
        event["payload"] = {
            "worker_id": worker_id,
            "command_event_id": command.command_event_id,
        }
        return event

    def progress_event(
        self,
        command: TaskCommand,
        *,
        status: str,
        stage: str,
        progress: int,
        message: str,
    ) -> dict[str, Any]:
        """生成 task.progress.updated 反馈事件。"""
        if status not in ALLOWED_PROGRESS_STATUSES:
            raise ValueError("status must be queued, running or waiting_user_input")
        if progress < 0 or progress > 100:
            raise ValueError("progress must be between 0 and 100")

        event = self._base_event(command, "task.progress.updated")
        event["payload"] = {
            "status": status,
            "stage": stage,
            "progress": progress,
            "message": sanitize_feedback_text(message),
        }
        return event

    def publish_progress(
        self,
        publisher: Any,
        command: TaskCommand,
        *,
        status: str,
        stage: str,
        progress: int,
        message: str,
    ) -> bool:
        """尽力发布进度事件，发布失败不阻断主流程。"""
        try:
            publisher.publish(
                self.progress_event(command, status=status, stage=stage, progress=progress, message=message)
            )
            return True
        except FeedbackPublishError:
            return False

    def result_event(self, command: TaskCommand, result: WorkerSuccessResult) -> dict[str, Any]:
        """生成 task.result.generated 反馈事件。"""
        event = self._base_event(command, "task.result.generated")
        event["payload"] = {
            "status": "success",
            "result_summary": result.result_summary,
            "output_files": [
                {
                    "object_key": artifact.object_key,
                    "file_type": artifact.file_type,
                    "file_purpose": artifact.file_purpose,
                    "mime_type": artifact.mime_type,
                    "file_size": artifact.file_size,
                    "display_name": artifact.display_name,
                }
                for artifact in result.output_files or []
            ],
        }
        return event

    def algorithm_problem_set_event(
        self,
        command: TaskCommand,
        result: WorkerSuccessResult,
    ) -> dict[str, Any]:
        """生成算法测试题单中间反馈事件。"""
        event = self._base_event(command, "algorithm_test.problem_set.generated")
        event["payload"] = {
            "status": "waiting_user_input",
            "stage": "waiting_user_input",
            "progress": 100,
            "message": "题单已生成，请完成 Codeforces 答题后发起提交检查",
            "result_summary": result.result_summary,
            "output_files": [
                {
                    "object_key": artifact.object_key,
                    "file_type": artifact.file_type,
                    "file_purpose": artifact.file_purpose,
                    "mime_type": artifact.mime_type,
                    "file_size": artifact.file_size,
                    "display_name": artifact.display_name,
                }
                for artifact in result.output_files or []
            ],
        }
        return event

    def mock_interview_intermediate_event(
        self,
        command: TaskCommand,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """生成模拟面试中间反馈事件。"""
        event = self._base_event(command, event_type)
        event_payload = {
            "status": payload.get("status", "waiting_user_input"),
            "stage": payload.get("stage", "waiting_user_input"),
            "progress": payload.get("progress", 100),
            **payload,
        }
        event_payload["message"] = sanitize_feedback_text(str(event_payload.get("message", "模拟面试状态已更新")))
        event["payload"] = event_payload
        return event

    def failed_event(self, command: TaskCommand, failure: WorkerFailure) -> dict[str, Any]:
        """生成 task.failed 反馈事件。"""
        event = self._base_event(command, "task.failed")
        payload = failure.to_payload()
        payload["user_message"] = sanitize_feedback_text(str(payload["user_message"]))
        payload["admin_summary"] = sanitize_feedback_text(payload.get("admin_summary"))
        event["payload"] = payload
        return event

    def _base_event(self, command: TaskCommand, event_type: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_id": self.event_id_supplier(),
            "event_type": event_type,
            "task_id": command.task_id,
            "task_type": command.task_type.value,
            "attempt": command.attempt,
            "trace_id": command.trace_id,
            "occurred_at": self.clock().isoformat().replace("+00:00", "Z"),
            "payload": {},
        }
