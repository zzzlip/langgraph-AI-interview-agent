"""简历优化 AI Gateway 编排。"""

from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.llm_config import chat_text, llm_request_kwargs
from ai_interview_worker.application.port.ai_gateway_port import AiChatRequest
from ai_interview_worker.application.resume_optimization.parser import OptimizationContext
from ai_interview_worker.application.resume_optimization.prompt import build_resume_optimization_messages
from ai_interview_worker.application.resume_optimization.quality_guard import validate_optimized_markdown
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode


def optimize_resume_markdown(
    ctx: AbilityContext,
    *,
    context: OptimizationContext,
    business_payload: dict[str, Any],
    min_length_ratio: float = 0.6,
) -> str:
    """通过 AI Gateway 生成优化后 Markdown。"""
    if ctx.ai_gateway is None:
        raise _ai_failure("AI 优化服务暂时不可用，请稍后重试")
    target_job = business_payload["target_job"]
    request = AiChatRequest(
        messages=build_resume_optimization_messages(
            target_job_title=target_job["title"],
            target_job_description=target_job["description"],
            context=context,
        ),
        temperature=0.3,
        max_tokens=4096,
        **llm_request_kwargs(ctx.command),
    )
    result = ctx.ai_gateway.chat_completion(request)
    if not result.success:
        raise _ai_failure("AI 优化服务暂时不可用，请稍后重试")
    markdown = _extract_markdown(result.data)
    if not markdown:
        markdown = _repair_once(ctx, context=context, target_job=target_job, invalid_markdown=str(result.data)[:800])
    try:
        validate_optimized_markdown(markdown, context, min_length_ratio=min_length_ratio)
    except ValueError:
        markdown = _repair_once(ctx, context=context, target_job=target_job, invalid_markdown=markdown[:800])
        try:
            validate_optimized_markdown(markdown, context, min_length_ratio=min_length_ratio)
        except ValueError as exc:
            raise _ai_failure("AI 优化结果格式异常，请稍后重试") from exc
    return markdown


def _extract_markdown(data: Any) -> str:
    markdown = _strip_code_fence(chat_text(data).strip())
    return markdown if markdown else ""


def _strip_code_fence(value: str) -> str:
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return value


def _repair_once(
    ctx: AbilityContext,
    *,
    context: OptimizationContext,
    target_job: dict[str, Any],
    invalid_markdown: str,
) -> str:
    if ctx.ai_gateway is None:
        raise _ai_failure("AI 优化服务暂时不可用，请稍后重试")
    request = AiChatRequest(
        messages=[
            {
                "role": "system",
                "content": (
                    "请修复优化后简历，只输出完整 Markdown。必须保留原始章节标题和顺序，"
                    "不得输出解释性前缀，不得改变原始语言。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"目标岗位：{target_job['title']}\n"
                    f"原始章节标题顺序：{', '.join(context.original_titles)}\n"
                    f"归一化章节类型：{', '.join(context.normalized_types)}\n"
                    f"原始语言：{context.language}\n"
                    f"受控简历上下文：\n{context.context_text}\n"
                    f"待修复内容：\n{invalid_markdown}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=4096,
        **llm_request_kwargs(ctx.command),
    )
    result = ctx.ai_gateway.chat_completion(request)
    if not result.success:
        raise _ai_failure("AI 优化服务暂时不可用，请稍后重试")
    markdown = _extract_markdown(result.data)
    if not markdown:
        raise _ai_failure("AI 优化结果格式异常，请稍后重试")
    return markdown


def _ai_failure(user_message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
            user_message=user_message,
            failed_stage="ai_optimization",
            retryable=True,
        )
    )
