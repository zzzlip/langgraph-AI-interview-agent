"""模拟面试题目生成。"""

from hashlib import sha256
from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.llm_config import (
    chat_json,
    llm_request_kwargs,
    private_llm_config,
    public_llm_config,
    resolve_task_llm_config,
)
from ai_interview_worker.application.port.ai_gateway_port import AiChatRequest
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode

QUESTION_TYPES = ("technical_fundamental", "project_experience", "business_scenario", "soft_skill")


def generate_interview_questions(
    ctx: AbilityContext,
    *,
    context: dict[str, str],
    question_count: int,
) -> list[dict[str, Any]]:
    """通过 LLM 基于简历和岗位上下文生成面试题。"""
    if ctx.ai_gateway is None:
        raise _ai_failure("模拟面试题目生成服务暂时不可用")
    result = ctx.ai_gateway.chat_completion(
        AiChatRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是资深技术面试官。请只返回 JSON 对象，不要 Markdown。"
                        "字段 questions 为数组，题量必须等于要求数量；每题字段为 sequence、question_type、title。"
                        "question_type 只能是 technical_fundamental、project_experience、"
                        "business_scenario、soft_skill。"
                        "题目必须基于简历和岗位动态生成，不能使用通用模板题。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"题量：{question_count}\n"
                        f"岗位标题：{context['target_job_title']}\n"
                        f"岗位摘要：{context['target_job_summary']}\n"
                        f"简历摘要：{context['resume_summary']}\n"
                        f"必须覆盖题型：{', '.join(QUESTION_TYPES)}"
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=2048,
            **llm_request_kwargs(ctx.command),
        )
    )
    if not result.success:
        raise _ai_failure("模拟面试题目生成服务暂时不可用")
    return _normalize_questions(ctx, chat_json(result.data), question_count)


def llm_config_payload(ctx: AbilityContext) -> dict[str, Any]:
    """返回模拟面试反馈中需要持久化的 LLM 配置。"""
    config = resolve_task_llm_config(ctx.command)
    return {
        "llm_config": private_llm_config(config),
        "public_llm_config": public_llm_config(config),
    }


def _normalize_questions(ctx: AbilityContext, payload: dict[str, Any], question_count: int) -> list[dict[str, Any]]:
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or len(raw_questions) != question_count:
        raise _ai_failure("模拟面试题目生成结果不完整")
    normalized: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    seen_titles: set[str] = set()
    for index, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            raise _ai_failure("模拟面试题目格式非法")
        question_type = str(item.get("question_type", "")).strip()
        title = str(item.get("title", "")).strip()
        if question_type not in QUESTION_TYPES or not title or title in seen_titles:
            raise _ai_failure("模拟面试题目格式非法")
        seen_types.add(question_type)
        seen_titles.add(title)
        normalized.append(
            {
                "question_id": _stable_id(ctx.command.task_id, ctx.command.attempt, index, title),
                "sequence": index + 1,
                "question_type": question_type,
                "title": title[:240],
                "source": "llm_resume_and_job_context",
                "status": "waiting_answer",
            }
        )
    if not {"technical_fundamental", "project_experience", "business_scenario"}.issubset(seen_types):
        raise _ai_failure("模拟面试题型覆盖不足")
    return normalized


def _stable_id(task_id: str, attempt: int, index: int, seed: str) -> str:
    digest = sha256(f"{task_id}:{attempt}:{index}:{seed}".encode()).hexdigest()
    return f"018f8a30-{digest[:4]}-7d31-9a60-{digest[4:16]}"


def _ai_failure(message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
            user_message=message,
            failed_stage="question_generation",
            retryable=True,
            admin_summary=message,
        )
    )
