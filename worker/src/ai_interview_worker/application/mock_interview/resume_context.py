"""模拟面试简历与岗位上下文构造。"""

from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode


def build_resume_job_context(ctx: AbilityContext, summary: dict[str, Any], *, max_chars: int = 12000) -> dict[str, str]:
    """读取简历文件并裁剪为题目生成上下文。"""
    resume_file = next((item for item in ctx.command.input_files or [] if item.file_purpose == "resume"), None)
    if resume_file is None:
        raise _failure("resume_reading", "缺少模拟面试简历文件")
    if ctx.minio_reader is None:
        raise _failure("resume_reading", "文件存储服务暂时不可用", retryable=True)
    read_result = ctx.minio_reader.get_object(resume_file.object_key)
    if not read_result.success or read_result.data is None:
        raise _failure("resume_reading", "简历文件暂时无法读取", retryable=True)
    resume_text = _decode_text(read_result.data)[:max_chars]
    target_job = summary.get("target_job")
    if not isinstance(target_job, dict):
        raise _failure("mock_interview_command", "目标岗位不能为空")
    title = str(target_job.get("title", "")).strip()
    description = str(target_job.get("description", "")).strip()
    if not title or not description:
        raise _failure("mock_interview_command", "目标岗位描述不能为空")
    return {
        "resume_text": resume_text,
        "resume_summary": _compact(resume_text, 500),
        "target_job_title": title,
        "target_job_description": description[:4000],
        "target_job_summary": _compact(description, 500),
    }


def _decode_text(data: bytes) -> str:
    text = data.decode("utf-8", errors="ignore").strip()
    if not text:
        raise _failure("resume_reading", "简历文件内容为空")
    return text


def _compact(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def _failure(stage: str, message: str, *, retryable: bool = False) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR if retryable else WorkerFailureCode.INPUT_ERROR,
            user_message=message,
            failed_stage=stage,
            retryable=retryable,
        )
    )
