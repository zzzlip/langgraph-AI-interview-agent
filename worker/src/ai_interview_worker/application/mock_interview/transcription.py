"""模拟面试回答转写。"""

from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.port.ai_gateway_port import AiTranscribeRequest
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode


def transcribe_answer(ctx: AbilityContext, answer_file: dict[str, Any]) -> str | None:
    """读取回答文件并调用 ASR，无法识别时返回 None。"""
    object_key = str(answer_file.get("object_key", "")).strip()
    file_type = str(answer_file.get("file_type", "mp3")).strip()
    if not object_key:
        raise _failure("answer_transcription", "回答文件引用不完整")
    if ctx.minio_reader is None:
        raise _failure("answer_transcription", "文件存储服务暂时不可用", retryable=True)
    if ctx.ai_gateway is None:
        raise _failure("answer_transcription", "语音转写服务暂时不可用", retryable=True)
    read_result = ctx.minio_reader.get_object(object_key)
    if not read_result.success or read_result.data is None:
        raise _failure("answer_transcription", "回答文件暂时无法读取", retryable=True)
    result = ctx.ai_gateway.transcribe_audio(AiTranscribeRequest(audio_data=read_result.data, audio_format=file_type))
    if not result.success:
        raise _failure("answer_transcription", "语音转写服务暂时不可用", retryable=True)
    transcript = str(result.data or "").strip()
    return transcript or None


def _failure(stage: str, message: str, *, retryable: bool = False) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR if retryable else WorkerFailureCode.INPUT_ERROR,
            user_message=message,
            failed_stage=stage,
            retryable=retryable,
        )
    )
