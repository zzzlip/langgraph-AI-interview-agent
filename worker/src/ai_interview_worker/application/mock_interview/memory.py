"""模拟面试任务内记忆更新。"""

from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.llm_config import chat_json, llm_request_kwargs
from ai_interview_worker.application.port.ai_gateway_port import AiChatRequest
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode


def compress_task_memory(
    ctx: AbilityContext,
    *,
    input_summary: dict[str, Any],
    transcript: str,
    draft_memory: dict[str, Any],
) -> dict[str, Any]:
    """使用 LLM 将本轮回答压缩进任务内结构化记忆。"""
    if ctx.ai_gateway is None:
        raise _memory_failure("任务内记忆压缩服务暂时不可用")
    result = ctx.ai_gateway.chat_completion(
        AiChatRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你负责模拟面试任务内短期记忆压缩。请只返回 JSON 对象，不要 Markdown。"
                        "字段：schema_version=1、question_memory 数组、candidate_signal_summary 对象、"
                        "follow_up_count 整数。question_memory 每项必须包含 question_id、round_id、"
                        "answer_signal、covered_points、missing_points。"
                        "禁止保存完整回答、完整 prompt、模型原始响应、object_key 或隐私明文。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"当前题：{input_summary.get('current_question_id')}\n"
                        f"当前轮：{input_summary.get('current_round_id')}\n"
                        f"现有记忆草稿：{draft_memory}\n"
                        f"本轮回答转写摘要：{transcript[:1200]}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1600,
            **llm_request_kwargs(ctx.command),
        )
    )
    if not result.success:
        raise _memory_failure("任务内记忆压缩服务暂时不可用")
    return _validate_memory(chat_json(result.data), input_summary)


def _validate_memory(payload: dict[str, Any], input_summary: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise _memory_failure("任务内记忆格式非法")
    question_memory = payload.get("question_memory")
    candidate_summary = payload.get("candidate_signal_summary")
    if not isinstance(question_memory, list) or not isinstance(candidate_summary, dict):
        raise _memory_failure("任务内记忆格式非法")
    normalized_entries: list[dict[str, Any]] = []
    for item in question_memory[-20:]:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or input_summary.get("current_question_id", "")).strip()
        round_id = str(item.get("round_id") or input_summary.get("current_round_id", "")).strip()
        answer_signal = " ".join(str(item.get("answer_signal", "")).split())[:180]
        if not question_id or not round_id or not answer_signal:
            continue
        normalized_entries.append(
            {
                "question_id": question_id,
                "round_id": round_id,
                "answer_signal": answer_signal,
                "covered_points": _string_list(item.get("covered_points"))[:6],
                "missing_points": _string_list(item.get("missing_points"))[:6],
            }
        )
    if not normalized_entries:
        raise _memory_failure("任务内记忆为空")
    return {
        "schema_version": 1,
        "question_memory": normalized_entries,
        "candidate_signal_summary": {
            "strengths": _string_list(candidate_summary.get("strengths"))[:5],
            "risks": _string_list(candidate_summary.get("risks"))[:5],
            "communication": str(candidate_summary.get("communication", "")).strip()[:160],
        },
        "follow_up_count": int(payload.get("follow_up_count", len(normalized_entries))),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _memory_failure(message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
            user_message=message,
            failed_stage="memory_update",
            retryable=True,
        )
    )
