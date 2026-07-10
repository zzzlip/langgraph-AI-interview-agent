"""模拟面试命令与结果 schema 校验。"""

from typing import Any

from ai_interview_worker.domain.command import TaskCommand
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.task_type import TaskType


def validate_mock_interview_command(command: TaskCommand, expected_action: str) -> dict[str, Any]:
    """校验模拟面试命令并返回 input_summary。"""
    if command.task_type != TaskType.MOCK_INTERVIEW:
        raise _failure("mock_interview_command", "任务类型不匹配")
    summary = command.input_summary or {}
    if summary.get("schema_version") != 1 or summary.get("action") != expected_action:
        raise _failure("mock_interview_command", "模拟面试命令参数不合法")
    return summary


def validate_questions_feedback(payload: dict[str, Any], *, question_count: int) -> None:
    """校验题目中间反馈 schema。"""
    questions = payload.get("questions")
    if payload.get("schema_version") != 1 or not isinstance(questions, list) or len(questions) != question_count:
        raise _failure("question_generation", "模拟面试题目生成结果不完整")
    seen_sequences: set[int] = set()
    seen_ids: set[str] = set()
    for item in questions:
        if not isinstance(item, dict):
            raise _failure("question_generation", "模拟面试题目格式非法")
        question_id = item.get("question_id")
        sequence = item.get("sequence")
        question_type = item.get("question_type")
        title = item.get("title")
        source = item.get("source")
        status = item.get("status")
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen_ids:
            raise _failure("question_generation", "模拟面试题目 ID 非法")
        if not isinstance(sequence, int) or sequence in seen_sequences:
            raise _failure("question_generation", "模拟面试题目序号非法")
        if question_type not in {"technical_fundamental", "project_experience", "business_scenario", "soft_skill"}:
            raise _failure("question_generation", "模拟面试题型非法")
        if not isinstance(title, str) or not title.strip():
            raise _failure("question_generation", "模拟面试题目内容为空")
        if not isinstance(source, str) or not source.strip() or status != "waiting_answer":
            raise _failure("question_generation", "模拟面试题目状态非法")
        seen_ids.add(question_id)
        seen_sequences.add(sequence)
    if payload.get("current_question_id") not in seen_ids:
        raise _failure("question_generation", "当前题目不在题单内")


def _failure(stage: str, message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.INPUT_ERROR,
            user_message=message,
            failed_stage=stage,
            retryable=False,
        )
    )
