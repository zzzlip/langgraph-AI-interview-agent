"""简历优化头像读取与校验。"""

from io import BytesIO

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.domain.command import InputFileRef
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_PREFIX = b"\xff\xd8"


def read_optional_avatar(ctx: AbilityContext, avatar_file: InputFileRef | None) -> bytes | None:
    """读取可选头像；缺失时返回 None。"""
    if avatar_file is None:
        return None
    if ctx.minio_reader is None:
        raise _avatar_failure(
            "文件存储服务暂时不可用，请稍后重试",
            WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
            retryable=True,
        )
    read_result = ctx.minio_reader.get_object(avatar_file.object_key)
    if not read_result.success or not read_result.data:
        raise _avatar_failure("头像文件无法读取，请重新上传后再试", WorkerFailureCode.INPUT_ERROR, retryable=False)
    if not _is_supported_image(read_result.data):
        raise _avatar_failure(
            "头像文件无法解析，请重新上传 PNG 或 JPG 图片",
            WorkerFailureCode.INPUT_ERROR,
            retryable=False,
        )
    return read_result.data


def _is_supported_image(data: bytes) -> bool:
    if not (data.startswith(PNG_SIGNATURE) or data.startswith(JPEG_PREFIX)):
        return False
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception:
        return False
    return True


def _avatar_failure(user_message: str, failure_code: WorkerFailureCode, *, retryable: bool) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=failure_code,
            user_message=user_message,
            failed_stage="resume_reading",
            retryable=retryable,
        )
    )
