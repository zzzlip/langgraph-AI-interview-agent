"""模拟面试报告生成。"""

from datetime import datetime, timezone
from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.port.file_ports import build_artifact_object_key
from ai_interview_worker.domain.artifact import OutputArtifact
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode


def render_report_pdf(result_summary: dict[str, Any]) -> bytes:
    """生成可下载的模拟面试报告 bytes。"""
    lines = ["Mock Interview Report", f"Overall Score: {result_summary['overall_score']}"]
    for item in result_summary["question_analyses"]:
        lines.append(f"- {item['title']}: {item['score']} / {item['answer_summary']}")
    body = "\n".join(lines).encode("utf-8")
    return b"%PDF-1.4\n" + body + b"\n%%EOF"


def write_report_artifact(ctx: AbilityContext, report_bytes: bytes) -> OutputArtifact:
    """写入模拟面试报告产物。"""
    display_name = "模拟面试报告.pdf"
    object_key = build_artifact_object_key(
        ctx.command.task_id, ctx.command.attempt, "interview_report", display_name
    )
    if ctx.artifact_writer is not None:
        result = ctx.artifact_writer.put_object(
            object_key=object_key,
            data=report_bytes,
            mime_type="application/pdf",
            display_name=display_name,
            clock=ctx.clock or _utc_now,
        )
        if not result.success or result.artifact is None:
            raise _report_failure()
        return result.artifact
    return OutputArtifact(
        object_key=object_key,
        file_type="pdf",
        file_purpose="interview_report",
        mime_type="application/pdf",
        file_size=len(report_bytes),
        display_name=display_name,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _report_failure() -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.ARTIFACT_WRITE_ERROR,
            user_message="模拟面试报告生成失败，请稍后重试",
            failed_stage="report_generation",
            retryable=True,
        )
    )
