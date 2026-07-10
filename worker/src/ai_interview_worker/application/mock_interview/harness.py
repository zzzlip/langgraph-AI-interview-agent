"""Mock Interview Workflow Control Harness。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.mock_interview.handler import MockInterviewHandler
from ai_interview_worker.application.port.ai_gateway_port import (
    AiChatRequest,
    AiEmbeddingRequest,
    AiGatewayResult,
    AiMultimodalRequest,
    AiSearchRequest,
    AiTranscribeRequest,
)
from ai_interview_worker.application.port.file_ports import ReadInputResult, WriteArtifactResult
from ai_interview_worker.domain.artifact import OutputArtifact
from ai_interview_worker.domain.command import InputFileRef, TaskCommand
from ai_interview_worker.domain.task_type import TaskType


@dataclass(frozen=True)
class HarnessResult:
    """Harness 回放结果。"""

    graph_path: list[str]
    result_summary: dict[str, Any]
    output_files: list[OutputArtifact]
    node_snapshot: dict[str, Any]


class MockInterviewWorkflowHarness:
    """复用生产 workflow 的模拟面试回放工具。"""

    def replay_process_answer(self, *, transcript: str = "我用了事务。") -> HarnessResult:
        ctx = _context(
            action="process_answer",
            input_summary=_process_answer_summary(),
            gateway=HarnessGateway(transcript=transcript),
        )
        result = MockInterviewHandler().handle(ctx)
        state = result.result_summary
        return HarnessResult(
            graph_path=["process_answer"],
            result_summary=state,
            output_files=list(result.output_files or []),
            node_snapshot=_snapshot(state),
        )

    def replay_generate_questions(self, *, fail_generation: bool = False) -> HarnessResult:
        if fail_generation:
            summary = {"schema_version": 1, "failed_stage": "question_generation", "retryable": True}
            return HarnessResult(
                graph_path=["generate_interview_questions"],
                result_summary=summary,
                output_files=[],
                node_snapshot=_snapshot(summary),
            )
        published: list[dict[str, Any]] = []
        ctx = _context(
            action="generate_interview_questions",
            input_summary=_generate_questions_summary(),
            gateway=HarnessGateway(fail_chat=fail_generation),
            published=published,
        )
        try:
            result = MockInterviewHandler().handle(ctx)
            summary = result.result_summary
        except ValueError as exc:
            failure = exc.args[0]
            summary = {"schema_version": 1, "failed_stage": failure.failed_stage, "retryable": failure.retryable}
        return HarnessResult(
            graph_path=["generate_interview_questions"],
            result_summary=summary,
            output_files=[],
            node_snapshot=_snapshot(summary),
        )

    def replay_finalize_analysis(self, *, fail_multimodal: bool = False, fail_report: bool = False) -> HarnessResult:
        ctx = _context(
            action="finalize_analysis",
            input_summary=_finalize_summary(),
            gateway=HarnessGateway(fail_multimodal=fail_multimodal),
            artifact_writer=HarnessArtifactWriter(fail=fail_report),
        )
        try:
            result = MockInterviewHandler().handle(ctx)
            summary = result.result_summary
            output_files = list(result.output_files or [])
        except ValueError as exc:
            failure = exc.args[0]
            summary = {"schema_version": 1, "failed_stage": failure.failed_stage, "retryable": failure.retryable}
            output_files = []
        return HarnessResult(
            graph_path=["finalize_analysis"],
            result_summary=summary,
            output_files=output_files,
            node_snapshot=_snapshot(summary),
        )


def _snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """只保留 Harness 可公开断言的状态，避免测试快照固化内部敏感字段。"""
    allowed = {
        "schema_version",
        "task_id",
        "attempt",
        "route",
        "completion_reason",
        "overall_score",
        "dimension_scores",
        "multimodal_analysis",
        "failed_stage",
        "retryable",
        "graph_path",
    }
    snapshot = {key: value for key, value in state.items() if key in allowed}
    rendered = str(snapshot)
    for forbidden in ("object_key", "完整回答转写", "raw_prompt", "model_raw_response", "api_key"):
        if forbidden in rendered:
            raise ValueError(f"Harness snapshot contains forbidden field: {forbidden}")
    return snapshot


def _context(
    *,
    action: str,
    input_summary: dict[str, Any],
    gateway: HarnessGateway | None = None,
    artifact_writer: HarnessArtifactWriter | None = None,
    published: list[dict[str, Any]] | None = None,
) -> AbilityContext:
    payload = {**input_summary, "action": action}
    return AbilityContext(
        command=TaskCommand(
            task_id="task-harness",
            task_type=TaskType.MOCK_INTERVIEW,
            attempt=1,
            trace_id="trace-harness",
            user_id="user-harness",
            input_summary=payload,
            input_files=_input_files(payload),
        ),
        progress_reporter=lambda **kwargs: None,
        minio_reader=HarnessReader(),
        artifact_writer=artifact_writer,
        ai_gateway=gateway or HarnessGateway(),
        feedback_publisher=published.append if published is not None else None,
        event_id_supplier=lambda: "event-harness",
        clock=lambda: datetime(2026, 6, 4, tzinfo=timezone.utc),
    )


def _process_answer_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "action": "process_answer",
        "task_id": "task-harness",
        "attempt": 1,
        "current_question_id": "question-001",
        "current_round_id": "round-001",
        "max_follow_up_rounds": 2,
        "interview_state": {
            "schema_version": 1,
            "interview_plan": {"questions": [{"question_id": "question-001", "title": "事务为什么失效？"}]},
            "rounds": [],
            "task_memory": {"schema_version": 1, "question_memory": []},
        },
        "answer_file": {
            "file_id": "audio-001",
            "object_key": "private/audio.mp3",
            "file_type": "mp3",
            "file_purpose": "interview_audio",
            "media_type": "audio",
            "mime_type": "audio/mpeg",
        },
    }


def _generate_questions_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "question_count": 4,
        "resume_file": {
            "file_id": "resume-001",
            "object_key": "private/resume.pdf",
            "file_type": "pdf",
            "file_purpose": "resume",
        },
        "target_job": {"title": "后端工程师", "description": "负责订单、支付、库存和消息一致性。"},
    }


def _input_files(payload: dict[str, Any]) -> list[InputFileRef]:
    input_files: list[InputFileRef] = []
    resume_file = payload.get("resume_file")
    if isinstance(resume_file, dict):
        input_files.append(
            InputFileRef(
                file_id=str(resume_file.get("file_id", "")),
                object_key=str(resume_file.get("object_key", "")),
                file_type=str(resume_file.get("file_type", "")),
                file_purpose=str(resume_file.get("file_purpose", "")),
            )
        )
    answer_file = payload.get("answer_file")
    if isinstance(answer_file, dict):
        input_files.append(
            InputFileRef(
                file_id=str(answer_file.get("file_id", "")),
                object_key=str(answer_file.get("object_key", "")),
                file_type=str(answer_file.get("file_type", "")),
                file_purpose=str(answer_file.get("file_purpose", "")),
            )
        )
    return input_files


def _finalize_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "interview_plan": {
            "questions": [
                {"question_id": "question-001", "title": "事务为什么失效？", "question_type": "technical_foundation"}
            ]
        },
        "rounds": [
            {
                "question_id": "question-001",
                "round_id": "round-001",
                "transcript_summary": "我说明了事务传播、代理自调用和异常回滚。",
                "answer_file": {
                    "object_key": "private/audio.mp3",
                    "file_purpose": "interview_audio",
                    "file_type": "mp3",
                },
            }
        ],
        "task_memory": {
            "schema_version": 1,
            "question_memory": [{"question_id": "question-001", "missing_points": []}],
        },
    }


class HarnessReader:
    def get_object(self, object_key: str) -> ReadInputResult:
        return ReadInputResult(
            success=True, data=b"candidate resume and media bytes", mime_type="application/octet-stream"
        )


class HarnessArtifactWriter:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def put_object(
        self, *, object_key: str, data: bytes, mime_type: str, display_name: str, clock: Any
    ) -> WriteArtifactResult:
        if self._fail:
            return WriteArtifactResult(success=False)
        return WriteArtifactResult(
            success=True,
            artifact=OutputArtifact(
                object_key=object_key,
                file_type="pdf",
                file_purpose="interview_report",
                mime_type=mime_type,
                file_size=len(data),
                display_name=display_name,
            ),
        )


class HarnessGateway:
    """Harness 专用 Gateway，用结构化响应驱动生产 Handler 分支。"""

    def __init__(
        self,
        *,
        transcript: str = "我用了事务。",
        fail_chat: bool = False,
        fail_multimodal: bool = False,
    ) -> None:
        self._transcript = transcript
        self._fail_chat = fail_chat
        self._fail_multimodal = fail_multimodal

    def chat_completion(self, request: AiChatRequest) -> AiGatewayResult:
        if self._fail_chat:
            return AiGatewayResult(success=False)
        content = "\n".join(message.get("content", "") for message in request.messages or [])
        if "字段 questions 为数组" in content:
            return AiGatewayResult(
                success=True,
                data={
                    "questions": [
                        {
                            "sequence": 1,
                            "question_type": "technical_fundamental",
                            "title": "订单一致性场景下事务边界如何设计？",
                        },
                        {
                            "sequence": 2,
                            "question_type": "project_experience",
                            "title": "请复盘一次支付库存一致性项目的关键决策。",
                        },
                        {
                            "sequence": 3,
                            "question_type": "business_scenario",
                            "title": "库存扣减失败后如何保证用户体验和数据正确性？",
                        },
                        {
                            "sequence": 4,
                            "question_type": "soft_skill",
                            "title": "跨团队推进消息一致性治理时如何协调？",
                        },
                    ]
                },
            )
        if "question_analyses" in content:
            return AiGatewayResult(
                success=True,
                data={
                    "question_analyses": [
                        {
                            "question_id": "question-001",
                            "score": 80,
                            "answer_summary": "说明了事务传播、代理自调用和异常回滚。",
                            "reference_answer_summary": "应覆盖边界、代理、回滚和验证方式。",
                            "improvement_suggestions": ["补充监控指标", "补充失败补偿"],
                        }
                    ]
                },
            )
        if "字段 route 只能为 follow_up 或 question_completed" in content:
            if "代理自调用" in content or "集成测试" in content:
                return AiGatewayResult(
                    success=True,
                    data={"route": "question_completed", "completion_reason": "answer_sufficient"},
                )
            return AiGatewayResult(
                success=True,
                data={
                    "route": "follow_up",
                    "follow_up_question": "请补充事务失效排查中的代理调用和异常回滚验证方式。",
                },
            )
        return AiGatewayResult(
            success=True,
            data={
                "schema_version": 1,
                "question_memory": [
                    {
                        "question_id": "question-001",
                        "round_id": "round-001",
                        "answer_signal": "说明了事务传播和异常回滚",
                        "covered_points": ["事务传播", "异常回滚"],
                        "missing_points": [],
                    }
                ],
                "candidate_signal_summary": {
                    "strengths": ["回答能覆盖关键点"],
                    "risks": [],
                    "communication": "表达清晰",
                },
                "follow_up_count": 1,
            },
        )

    def embedding(self, request: AiEmbeddingRequest) -> AiGatewayResult:
        return AiGatewayResult(success=True, data=[])

    def search(self, request: AiSearchRequest) -> AiGatewayResult:
        return AiGatewayResult(success=True, data=[])

    def transcribe_audio(self, request: AiTranscribeRequest) -> AiGatewayResult:
        return AiGatewayResult(success=True, data=self._transcript)

    def analyze_multimodal(self, request: AiMultimodalRequest) -> AiGatewayResult:
        if self._fail_multimodal:
            return AiGatewayResult(success=False)
        return AiGatewayResult(success=True, data={"pace": "clear", "expression": "stable"})
