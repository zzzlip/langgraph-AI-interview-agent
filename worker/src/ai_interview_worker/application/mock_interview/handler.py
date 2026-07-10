"""模拟面试业务 Handler。"""

from datetime import datetime, timezone
from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext, AbilityHandler
from ai_interview_worker.application.feedback_service import FeedbackService
from ai_interview_worker.application.mock_interview.answer_analyzer import analyze_completed_questions_with_llm
from ai_interview_worker.application.mock_interview.follow_up import decide_and_generate_follow_up
from ai_interview_worker.application.mock_interview.memory import compress_task_memory
from ai_interview_worker.application.mock_interview.multimodal import analyze_multimodal_signals
from ai_interview_worker.application.mock_interview.question_generator import (
    generate_interview_questions,
    llm_config_payload,
)
from ai_interview_worker.application.mock_interview.report import render_report_pdf, write_report_artifact
from ai_interview_worker.application.mock_interview.resume_context import build_resume_job_context
from ai_interview_worker.application.mock_interview.schema import (
    validate_mock_interview_command,
    validate_questions_feedback,
)
from ai_interview_worker.application.mock_interview.transcription import transcribe_answer
from ai_interview_worker.application.mock_interview.workflow import run_process_answer_workflow
from ai_interview_worker.domain.artifact import OutputArtifact
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.result import WorkerSuccessResult


class MockInterviewHandler(AbilityHandler):
    """模拟面试正式 Handler。"""

    def handle(self, ctx: AbilityContext) -> WorkerSuccessResult:
        action = (ctx.command.input_summary or {}).get("action")
        if action == "generate_interview_questions":
            return self._generate_interview_questions(ctx)
        if action == "process_answer":
            return self._process_answer(ctx)
        if action == "finalize_analysis":
            return self._finalize_analysis(ctx)
        raise ValueError(
            WorkerFailure(
                failure_code=WorkerFailureCode.INPUT_ERROR,
                user_message="不支持的模拟面试动作",
                failed_stage="mock_interview_command",
                retryable=False,
            )
        )

    def _generate_interview_questions(self, ctx: AbilityContext) -> WorkerSuccessResult:
        summary = validate_mock_interview_command(ctx.command, "generate_interview_questions")
        question_count = int(summary.get("question_count", 6))
        if question_count < 4 or question_count > 8:
            raise ValueError(
                WorkerFailure(
                    failure_code=WorkerFailureCode.INPUT_ERROR,
                    user_message="模拟面试题量配置非法",
                    failed_stage="mock_interview_command",
                    retryable=False,
                )
            )
        ctx.progress_reporter(stage="resume_context", progress=20, message="正在读取模拟面试上下文")
        context = build_resume_job_context(ctx, summary)
        ctx.progress_reporter(stage="question_generation", progress=70, message="正在生成模拟面试题目")
        questions = generate_interview_questions(ctx, context=context, question_count=question_count)
        llm_payload = llm_config_payload(ctx)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task_id": ctx.command.task_id,
            "attempt": ctx.command.attempt,
            "status": "waiting_user_input",
            "stage": "waiting_user_input",
            "progress": 100,
            "message": "面试题已生成，请回答第一题",
            "questions": questions,
            "current_question_id": questions[0]["question_id"],
            "llm_config": llm_payload["llm_config"],
            "public_llm_config": llm_payload["public_llm_config"],
            "memory": {
                "schema_version": 1,
                "question_memory": [],
                "candidate_signal_summary": {
                    "strengths": [],
                    "risks": [],
                    "communication": "尚未形成表现摘要",
                },
            },
        }
        validate_questions_feedback(payload, question_count=question_count)
        _publish_questions_feedback(ctx, payload)
        # result_summary 可能进入任务详情和 SSE，必须去掉 Worker 内部继续流转才需要的私密 llm_config。
        public_payload = dict(payload)
        public_payload.pop("llm_config", None)
        return WorkerSuccessResult(result_summary=public_payload, output_files=[])

    def _process_answer(self, ctx: AbilityContext) -> WorkerSuccessResult:
        summary = validate_mock_interview_command(ctx.command, "process_answer")
        answer_file = summary.get("answer_file")
        if not isinstance(answer_file, dict):
            raise ValueError(
                WorkerFailure(
                    failure_code=WorkerFailureCode.INPUT_ERROR,
                    user_message="回答文件引用不完整",
                    failed_stage="answer_transcription",
                    retryable=False,
                )
            )
        ctx.progress_reporter(stage="answer_transcription", progress=40, message="正在转写模拟面试回答")
        transcript = transcribe_answer(ctx, answer_file)
        if transcript is None:
            payload = _needs_resubmit_payload(ctx, summary)
            _publish_intermediate(ctx, "mock_interview.answer.needs_resubmit", payload)
            return WorkerSuccessResult(result_summary=payload, output_files=[])
        workflow_state = run_process_answer_workflow(summary, transcript)
        if workflow_state.get("safety_blocked") is True:
            payload = _question_completed_payload(
                ctx,
                summary,
                transcript,
                {
                    "completion_reason": str(workflow_state.get("safety_reason", "safety_blocked_follow_up")),
                    "task_memory": workflow_state.get("task_memory", {"schema_version": 1}),
                },
            )
            _publish_intermediate(ctx, "mock_interview.question.completed", payload)
            return WorkerSuccessResult(result_summary=payload, output_files=[])
        task_memory = compress_task_memory(
            ctx,
            input_summary=summary,
            transcript=transcript,
            draft_memory=workflow_state.get("task_memory", {"schema_version": 1}),
        )
        decision = decide_and_generate_follow_up(
            ctx,
            input_summary=summary,
            transcript=transcript,
            task_memory=task_memory,
        )
        workflow_state = {**workflow_state, **decision, "task_memory": task_memory}
        if workflow_state.get("route") == "follow_up":
            payload = _follow_up_payload(ctx, summary, workflow_state)
            _publish_intermediate(ctx, "mock_interview.follow_up.generated", payload)
            return WorkerSuccessResult(result_summary=payload, output_files=[])
        payload = _question_completed_payload(ctx, summary, transcript, workflow_state)
        _publish_intermediate(ctx, "mock_interview.question.completed", payload)
        return WorkerSuccessResult(result_summary=payload, output_files=[])

    def _finalize_analysis(self, ctx: AbilityContext) -> WorkerSuccessResult:
        summary = validate_mock_interview_command(ctx.command, "finalize_analysis")
        raw_interview_state = summary.get("interview_state")
        interview_state: dict[str, Any] = raw_interview_state if isinstance(raw_interview_state, dict) else summary
        plan = interview_state.get("interview_plan")
        rounds = interview_state.get("rounds")
        task_memory = interview_state.get("task_memory")
        if not isinstance(plan, dict) or not isinstance(rounds, list) or not isinstance(task_memory, dict):
            raise ValueError(
                WorkerFailure(
                    failure_code=WorkerFailureCode.INPUT_ERROR,
                    user_message="模拟面试最终分析上下文不完整",
                    failed_stage="finalize_analysis",
                    retryable=False,
                )
            )
        questions = plan.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError(
                WorkerFailure(
                    failure_code=WorkerFailureCode.INPUT_ERROR,
                    user_message="模拟面试题单为空，无法生成最终分析",
                    failed_stage="finalize_analysis",
                    retryable=False,
                )
            )
        ctx.progress_reporter(stage="answer_analysis", progress=45, message="正在生成模拟面试逐题分析")
        question_analyses = analyze_completed_questions_with_llm(
            ctx,
            questions=questions,
            rounds=rounds,
            task_memory=task_memory,
        )
        ctx.progress_reporter(stage="multimodal_analysis", progress=65, message="正在分析模拟面试多模态表现")
        multimodal_analysis = analyze_multimodal_signals(ctx, rounds)
        result_summary = _build_final_result_summary(
            task_id=ctx.command.task_id,
            attempt=ctx.command.attempt,
            question_analyses=question_analyses,
            multimodal_analysis=multimodal_analysis,
        )
        ctx.progress_reporter(stage="report_generation", progress=85, message="正在生成模拟面试报告")
        report_bytes = render_report_pdf(result_summary)
        artifact = write_report_artifact(ctx, report_bytes)
        result_summary["report_files"] = [_public_artifact(artifact)]
        ctx.progress_reporter(stage="completed", progress=100, message="模拟面试最终分析已完成")
        return WorkerSuccessResult(result_summary=result_summary, output_files=[artifact])


def _publish_questions_feedback(ctx: AbilityContext, payload: dict[str, Any]) -> None:
    _publish_intermediate(ctx, "mock_interview.questions.generated", payload)


def _publish_intermediate(ctx: AbilityContext, event_type: str, payload: dict[str, Any]) -> None:
    if ctx.feedback_publisher is None:
        return
    service = FeedbackService(
        event_id_supplier=ctx.event_id_supplier or (lambda: "mock-interview-questions-feedback"),
        clock=ctx.clock or _utc_now,
    )
    event = service.mock_interview_intermediate_event(
        ctx.command,
        event_type=event_type,
        payload=payload,
    )
    ctx.feedback_publisher(event)


def _needs_resubmit_payload(ctx: AbilityContext, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": ctx.command.task_id,
        "attempt": ctx.command.attempt,
        "status": "waiting_user_input",
        "stage": "waiting_user_input",
        "progress": 100,
        "message": "回答音频无法识别，请重新上传当前轮回答",
        "current_question_id": summary.get("current_question_id"),
        "current_round_id": summary.get("current_round_id"),
        "reason": "empty_transcript",
    }


def _follow_up_payload(ctx: AbilityContext, summary: dict[str, Any], workflow_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": ctx.command.task_id,
        "attempt": ctx.command.attempt,
        "status": "waiting_user_input",
        "stage": "waiting_user_input",
        "progress": 100,
        "message": "已生成追问，请继续回答",
        "route": "follow_up",
        "current_question_id": summary.get("current_question_id"),
        "current_round_id": summary.get("current_round_id"),
        "follow_up_question": workflow_state["follow_up_question"],
        "task_memory": workflow_state["task_memory"],
    }


def _question_completed_payload(
    ctx: AbilityContext,
    summary: dict[str, Any],
    transcript: str,
    workflow_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": ctx.command.task_id,
        "attempt": ctx.command.attempt,
        "status": "running",
        "stage": "answer_processed",
        "progress": 100,
        "message": "回答已转写，当前题处理完成",
        "route": "question_completed",
        "current_question_id": summary.get("current_question_id"),
        "current_round_id": summary.get("current_round_id"),
        "transcript_summary": transcript[:300],
        "completion_reason": _public_completion_reason(workflow_state),
        "task_memory": workflow_state.get("task_memory", {"schema_version": 1}),
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _public_completion_reason(workflow_state: dict[str, Any]) -> str:
    reason = str(workflow_state.get("completion_reason", "answer_transcribed"))
    if reason == "answer_sufficient":
        return "answer_transcribed"
    return reason


def _build_final_result_summary(
    *,
    task_id: str,
    attempt: int,
    question_analyses: list[dict[str, Any]],
    multimodal_analysis: dict[str, Any],
) -> dict[str, Any]:
    overall = _overall_score(question_analyses, multimodal_analysis)
    return {
        "schema_version": 1,
        "task_id": task_id,
        "attempt": attempt,
        "overall_score": overall,
        "dimension_scores": {
            "technical_depth": _dimension_score(question_analyses, "technical"),
            "project_experience": _dimension_score(question_analyses, "project"),
            "business_understanding": _dimension_score(question_analyses, "business"),
            "communication": min(100, overall + 3),
            "multimodal": 78 if multimodal_analysis.get("status") == "ok" else 62,
        },
        "question_analyses": question_analyses,
        "multimodal_analysis": multimodal_analysis,
        "candidate_performance_summary": _candidate_summary(question_analyses),
        "improvement_plan": _improvement_plan(question_analyses),
        "report_files": [],
    }


def _overall_score(question_analyses: list[dict[str, Any]], multimodal_analysis: dict[str, Any]) -> int:
    if not question_analyses:
        return 0
    base = sum(int(item.get("score", 0)) for item in question_analyses) // len(question_analyses)
    if multimodal_analysis.get("status") == "degraded":
        return max(0, base - 3)
    return min(100, base + 2)


def _dimension_score(question_analyses: list[dict[str, Any]], marker: str) -> int:
    selected = [
        int(item.get("score", 0))
        for item in question_analyses
        if marker in str(item.get("question_type", "")) or marker in str(item.get("title", "")).lower()
    ]
    if not selected:
        selected = [int(item.get("score", 0)) for item in question_analyses]
    return sum(selected) // len(selected) if selected else 0


def _candidate_summary(question_analyses: list[dict[str, Any]]) -> str:
    strong = [item["title"] for item in question_analyses if int(item.get("score", 0)) >= 75]
    if strong:
        return f"候选人在{strong[0][:40]}等问题上表达较完整。"
    return "候选人已完成模拟面试，但关键论证细节仍需补强。"


def _improvement_plan(question_analyses: list[dict[str, Any]]) -> list[str]:
    suggestions: list[str] = []
    for item in question_analyses:
        values = item.get("improvement_suggestions")
        if isinstance(values, list):
            suggestions.extend(str(value) for value in values)
    return suggestions[:6] or ["补充项目背景、关键决策、风险处理和验证结果"]


def _public_artifact(artifact: OutputArtifact) -> dict[str, Any]:
    return {
        "file_type": artifact.file_type,
        "file_purpose": artifact.file_purpose,
        "mime_type": artifact.mime_type,
        "file_size": artifact.file_size,
        "display_name": artifact.display_name,
    }
