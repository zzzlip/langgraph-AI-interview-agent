"""模拟面试追问判断与生成。"""

from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext
from ai_interview_worker.application.llm_config import chat_json, llm_request_kwargs
from ai_interview_worker.application.port.ai_gateway_port import AiChatRequest
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode


def decide_and_generate_follow_up(
    ctx: AbilityContext,
    *,
    input_summary: dict[str, Any],
    transcript: str,
    task_memory: dict[str, Any],
) -> dict[str, Any]:
    """使用 LLM 判断是否追问，并生成追问内容或完成原因。"""
    if ctx.ai_gateway is None:
        raise _follow_up_failure("模拟面试追问服务暂时不可用")
    current_question = _current_question(input_summary)
    result = ctx.ai_gateway.chat_completion(
        AiChatRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是模拟面试官。请只返回 JSON 对象，不要 Markdown。"
                        "字段 route 只能为 follow_up 或 question_completed。"
                        "route=follow_up 时必须返回 follow_up_question；"
                        "route=question_completed 时必须返回 completion_reason。"
                        "追问必须基于当前题、当前回答和任务内记忆，不得泄露参考答案、标准答案或推理链。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"当前题：{current_question}\n"
                        f"回答转写摘要：{transcript[:1200]}\n"
                        f"任务内记忆：{task_memory}\n"
                        f"最大追问轮数：{input_summary.get('max_follow_up_rounds', 2)}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1000,
            **llm_request_kwargs(ctx.command),
        )
    )
    if not result.success:
        raise _follow_up_failure("模拟面试追问服务暂时不可用")
    return _validate_decision(chat_json(result.data), task_memory)


def _current_question(input_summary: dict[str, Any]) -> str:
    interview_state = input_summary.get("interview_state")
    plan = interview_state.get("interview_plan") if isinstance(interview_state, dict) else None
    questions = plan.get("questions") if isinstance(plan, dict) else None
    current_question_id = input_summary.get("current_question_id")
    if isinstance(questions, list):
        for item in questions:
            if isinstance(item, dict) and item.get("question_id") == current_question_id:
                return str(item.get("title", "当前题目"))[:240]
    return "当前题目"


def _validate_decision(payload: dict[str, Any], task_memory: dict[str, Any]) -> dict[str, Any]:
    route = str(payload.get("route", "")).strip()
    if route == "follow_up":
        question = str(payload.get("follow_up_question", "")).strip()
        if not question or any(marker in question for marker in ("参考答案", "标准答案", "推理链")):
            raise _follow_up_failure("模拟面试追问内容非法")
        return {"route": route, "follow_up_question": question[:240], "task_memory": task_memory}
    if route == "question_completed":
        reason = str(payload.get("completion_reason", "answer_sufficient")).strip() or "answer_sufficient"
        return {"route": route, "completion_reason": reason[:120], "task_memory": task_memory}
    raise _follow_up_failure("模拟面试追问决策格式非法")


def _follow_up_failure(message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
            user_message=message,
            failed_stage="follow_up_generation",
            retryable=True,
        )
    )
