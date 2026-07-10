"""简历优化业务 Handler。"""

from datetime import datetime, timezone

from ai_interview_worker.application.ability_handler import AbilityContext, AbilityHandler
from ai_interview_worker.application.port.file_ports import build_artifact_object_key
from ai_interview_worker.application.resume_optimization.avatar import read_optional_avatar
from ai_interview_worker.application.resume_optimization.optimizer import optimize_resume_markdown
from ai_interview_worker.application.resume_optimization.parser import (
    ResumeDocument,
    build_optimization_context,
    parse_resume_document,
)
from ai_interview_worker.application.resume_optimization.pdf_renderer import render_resume_pdf
from ai_interview_worker.application.resume_optimization.schema import validate_resume_optimization_command
from ai_interview_worker.application.resume_optimization.summary_builder import build_result_summary
from ai_interview_worker.domain.artifact import OutputArtifact
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.result import WorkerSuccessResult


class ResumeOptimizationHandler(AbilityHandler):
    """简历优化业务 Handler。"""

    def handle(self, ctx: AbilityContext) -> WorkerSuccessResult:
        business_payload = validate_resume_optimization_command(ctx.command)
        resume_file = next(file for file in ctx.command.input_files or [] if file.file_purpose == "resume")
        avatar_file = next((file for file in ctx.command.input_files or [] if file.file_purpose == "avatar"), None)
        avatar_used = any(file.file_purpose == "avatar" for file in ctx.command.input_files or [])
        options = business_payload["optimization_options"]
        ctx.progress_reporter(stage="resume_reading", progress=10, message="正在读取简历")
        document = self._parse_resume(ctx, resume_file.object_key, resume_file.file_type)
        avatar = read_optional_avatar(ctx, avatar_file)
        context = build_optimization_context(document, max_chars=12000)
        ctx.progress_reporter(stage="ai_optimization", progress=45, message="正在优化简历内容")
        optimized_markdown = optimize_resume_markdown(ctx, context=context, business_payload=business_payload)
        ctx.progress_reporter(stage="pdf_generation", progress=85, message="正在生成优化后简历 PDF")
        pdf_bytes = self._render_pdf(optimized_markdown, avatar)
        ctx.progress_reporter(stage="completed", progress=100, message="简历优化已完成")
        result_summary = build_result_summary(
            document=document,
            avatar_used=avatar_used,
            prompt_version=options["prompt_version"],
            template=options["template"],
        )
        return WorkerSuccessResult(
            result_summary=result_summary,
            output_files=[_write_or_build_pdf_artifact(ctx, pdf_bytes)],
        )

    def _parse_resume(self, ctx: AbilityContext, object_key: str, file_type: str) -> ResumeDocument:
        if ctx.minio_reader is None:
            raise _failure(
                WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
                "文件存储服务暂时不可用，请稍后重试",
                retryable=True,
            )
        read_result = ctx.minio_reader.get_object(object_key)
        if not read_result.success or read_result.data is None:
            raise _failure(
                WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
                "简历文件暂时无法读取，请稍后重试",
                retryable=True,
            )
        try:
            return parse_resume_document(read_result.data, file_type)
        except ValueError as exc:
            raise _failure(
                WorkerFailureCode.INPUT_ERROR,
                "简历文件无法解析，请确认文件内容清晰后重新上传",
                retryable=False,
            ) from exc

    def _render_pdf(self, optimized_markdown: str, avatar: bytes | None) -> bytes:
        try:
            return render_resume_pdf(optimized_markdown, avatar)
        except Exception as exc:
            raise ValueError(
                WorkerFailure(
                    failure_code=WorkerFailureCode.INTERNAL_ERROR,
                    user_message="优化后简历 PDF 生成失败，请稍后重试",
                    failed_stage="pdf_generation",
                    retryable=True,
                )
            ) from exc


def _write_or_build_pdf_artifact(ctx: AbilityContext, pdf_bytes: bytes) -> OutputArtifact:
    display_name = "优化后简历.pdf"
    object_key = build_artifact_object_key(ctx.command.task_id, ctx.command.attempt, "optimized_resume", display_name)
    if ctx.artifact_writer is not None:
        result = ctx.artifact_writer.put_object(
            object_key=object_key,
            data=pdf_bytes,
            mime_type="application/pdf",
            display_name=display_name,
            clock=ctx.clock or _utc_now,
        )
        if not result.success or result.artifact is None:
            raise ValueError(
                WorkerFailure(
                    failure_code=WorkerFailureCode.ARTIFACT_WRITE_ERROR,
                    user_message="优化后简历 PDF 写入失败，请稍后重试",
                    failed_stage="pdf_generation",
                    retryable=True,
                )
            )
        return result.artifact
    return OutputArtifact(
        object_key=object_key,
        file_type="pdf",
        file_purpose="optimized_resume",
        mime_type="application/pdf",
        file_size=len(pdf_bytes),
        display_name=display_name,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _failure(failure_code: WorkerFailureCode, user_message: str, *, retryable: bool) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=failure_code,
            user_message=user_message,
            failed_stage="resume_reading",
            retryable=retryable,
        )
    )
