"""模拟面试多模态分析。"""

from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.port.ai_gateway_port import AiMultimodalRequest


def analyze_multimodal_signals(ctx: AbilityContext, rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """分析音频或视频表现，单项失败时降级记录。"""
    files: list[dict[str, Any]] = []
    for item in rounds:
        if not isinstance(item, dict):
            continue
        answer_file = item.get("answer_file")
        if isinstance(answer_file, dict):
            files.append(answer_file)
    if ctx.ai_gateway is None or ctx.minio_reader is None or not files:
        return {
            "schema_version": 1,
            "status": "degraded",
            "dimensions": {},
            "degraded": ["multimodal_input_unavailable"],
        }

    dimensions: dict[str, Any] = {}
    degraded: list[str] = []
    for file_ref in files:
        purpose = str(file_ref.get("file_purpose", ""))
        object_key = str(file_ref.get("object_key", ""))
        read_result = ctx.minio_reader.get_object(object_key)
        if not read_result.success or read_result.data is None:
            degraded.append(_safe_reason(purpose, "read_failed"))
            continue
        result = ctx.ai_gateway.analyze_multimodal(
            AiMultimodalRequest(
                audio_data=read_result.data if purpose == "interview_audio" else None,
                audio_format=str(file_ref.get("file_type", "mp3")).strip() or "mp3",
                video_data=read_result.data if purpose == "interview_video" else None,
                video_format=str(file_ref.get("file_type", "mp4")).strip() or "mp4",
                image_data=None,
                prompt="分析候选人回答的表达、语速、停顿和视频表现，只返回结构化摘要。",
            )
        )
        if not result.success:
            degraded.append(_safe_reason(purpose, "gateway_failed"))
            continue
        dimensions[purpose or "interview_media"] = (
            result.data if isinstance(result.data, dict) else {"summary": str(result.data)[:120]}
        )

    status = "ok" if dimensions and not degraded else "degraded"
    return {"schema_version": 1, "status": status, "dimensions": dimensions, "degraded": degraded}


def _safe_reason(purpose: str, reason: str) -> str:
    return f"{purpose or 'media'}:{reason}"
