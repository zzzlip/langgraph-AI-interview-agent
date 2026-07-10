"""简历优化命令与阶段契约校验。"""

from typing import Any

from ai_interview_worker.domain.command import InputFileRef, TaskCommand
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.task_type import TaskType

ALLOWED_PROGRESS_STAGES = (
    "resume_reading",
    "ai_optimization",
    "pdf_generation",
    "completed",
)
SUPPORTED_PROMPT_VERSION = "resume_optimization_v1"
SUPPORTED_TEMPLATE = "default_pdf_v1"


def validate_resume_optimization_command(command: TaskCommand) -> dict[str, Any]:
    """校验简历优化命令，并返回业务 payload。"""
    if command.task_type is not TaskType.RESUME_OPTIMIZATION:
        raise _input_failure("task_routing", "不支持的任务类型")

    resume_files = [file for file in command.input_files or [] if file.file_purpose == "resume"]
    if len(resume_files) != 1:
        raise _input_failure("input_check", "简历优化需要且只能包含一份简历文件")
    _validate_resume_file(resume_files[0])

    avatar_files = [file for file in command.input_files or [] if file.file_purpose == "avatar"]
    if len(avatar_files) > 1:
        raise _input_failure("input_check", "简历优化最多只能包含一份头像文件")
    if avatar_files:
        _validate_avatar_file(avatar_files[0])

    payload = command.input_data or {}
    business_payload = payload.get("business_payload")
    if not isinstance(business_payload, dict):
        raise _input_failure("input_check", "简历优化业务参数缺失")
    target_job = business_payload.get("target_job")
    if not isinstance(target_job, dict):
        raise _input_failure("input_check", "目标岗位信息缺失")
    title = target_job.get("title")
    description = target_job.get("description")
    if not isinstance(title, str) or not title.strip():
        raise _input_failure("input_check", "目标岗位名称不能为空")
    if not isinstance(description, str) or len(description.strip()) < 20:
        raise _input_failure("input_check", "目标岗位描述不能为空")

    options = business_payload.get("optimization_options")
    if not isinstance(options, dict):
        raise _input_failure("input_check", "简历优化选项缺失")
    prompt_version = options.get("prompt_version")
    template = options.get("template")
    if prompt_version != SUPPORTED_PROMPT_VERSION:
        raise _internal_failure("input_check", "简历优化 Prompt 配置暂不可用")
    if template != SUPPORTED_TEMPLATE:
        raise _internal_failure("input_check", "简历优化模板配置暂不可用")

    return business_payload


def assert_progress_stage(stage: str) -> None:
    """校验简历优化进度阶段。"""
    if stage not in ALLOWED_PROGRESS_STAGES:
        raise ValueError("unsupported resume optimization progress stage")


def _validate_resume_file(resume_file: InputFileRef) -> None:
    if resume_file.file_type not in {"pdf", "doc", "docx"}:
        raise _input_failure("input_check", "简历文件类型不支持")
    if not resume_file.object_key.strip():
        raise _input_failure("input_check", "简历文件引用无效")


def _validate_avatar_file(avatar_file: InputFileRef) -> None:
    if avatar_file.file_type not in {"png", "jpg", "jpeg"}:
        raise _input_failure("input_check", "头像文件类型不支持")
    if not avatar_file.object_key.strip():
        raise _input_failure("input_check", "头像文件引用无效")


def _input_failure(stage: str, message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.INPUT_ERROR,
            user_message=message,
            failed_stage=stage,
            retryable=False,
        )
    )


def _internal_failure(stage: str, message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.INTERNAL_ERROR,
            user_message=message,
            failed_stage=stage,
            retryable=True,
        )
    )
