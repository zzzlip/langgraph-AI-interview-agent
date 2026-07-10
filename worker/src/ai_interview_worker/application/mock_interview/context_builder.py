"""模拟面试追问上下文构造。"""

from typing import Any


def build_follow_up_context(input_summary: dict[str, Any], transcript: str) -> dict[str, Any]:
    """只构造当前题上下文，避免串题和泄露标准答案。"""
    raw_interview_state = input_summary.get("interview_state")
    interview_state: dict[str, Any] = raw_interview_state if isinstance(raw_interview_state, dict) else {}
    current_question_id = str(input_summary.get("current_question_id", ""))
    current_round_id = str(input_summary.get("current_round_id", ""))
    current_title = _current_question_title(interview_state, current_question_id)
    rounds = _current_question_rounds(interview_state, current_question_id)
    if _contains_forbidden_context(interview_state, current_question_id):
        return {
            "current_question_id": current_question_id,
            "current_round_id": current_round_id,
            "current_question_title": current_title,
            "transcript_summary": transcript[:500],
            "task_memory": interview_state.get("task_memory", {"schema_version": 1}),
            "context_snapshot": {},
            "safety_blocked": True,
            "safety_reason": "unsafe_cross_question_context",
        }
    return {
        "current_question_id": current_question_id,
        "current_round_id": current_round_id,
        "current_question_title": current_title,
        "transcript_summary": transcript[:500],
        "task_memory": interview_state.get("task_memory", {"schema_version": 1}),
        "context_snapshot": {
            "schema_version": 1,
            "current_question": {
                "question_id": current_question_id,
                "title": current_title,
            },
            "current_question_rounds": rounds,
            "question_memory": _question_memory(interview_state, current_question_id),
            "resume_summary": str(input_summary.get("resume_summary", ""))[:600],
            "target_job_summary": str(input_summary.get("target_job_summary", ""))[:600],
            "transcript_summary": transcript[:500],
        },
        "safety_blocked": False,
    }


def _current_question_title(interview_state: dict[str, Any], question_id: str) -> str:
    plan = interview_state.get("interview_plan")
    questions = plan.get("questions") if isinstance(plan, dict) else []
    if isinstance(questions, list):
        for item in questions:
            if isinstance(item, dict) and item.get("question_id") == question_id:
                return str(item.get("title", "当前题目"))[:300]
    return "当前题目"


def _current_question_rounds(interview_state: dict[str, Any], question_id: str) -> list[dict[str, str]]:
    rounds = interview_state.get("rounds")
    if not isinstance(rounds, list):
        return []
    safe_rounds: list[dict[str, str]] = []
    for item in rounds:
        if not isinstance(item, dict) or item.get("question_id") != question_id:
            continue
        safe_rounds.append(
            {
                "round_id": str(item.get("round_id", ""))[:80],
                "answer_summary": str(item.get("answer_summary", item.get("transcript_summary", "")))[:300],
                "follow_up_question": str(item.get("follow_up_question", ""))[:300],
            }
        )
    return safe_rounds[-6:]


def _question_memory(interview_state: dict[str, Any], question_id: str) -> list[dict[str, str]]:
    memory = interview_state.get("task_memory")
    entries = memory.get("question_memory") if isinstance(memory, dict) else []
    if not isinstance(entries, list):
        return []
    return [
        {
            "round_id": str(item.get("round_id", ""))[:80],
            "answer_signal": str(item.get("answer_signal", ""))[:160],
        }
        for item in entries
        if isinstance(item, dict) and item.get("question_id") == question_id
    ][-6:]


def _contains_forbidden_context(interview_state: dict[str, Any], current_question_id: str) -> bool:
    forbidden_markers = ("参考答案", "标准答案", "推理链", "raw_prompt", "model_raw_response")
    rounds = interview_state.get("rounds")
    if isinstance(rounds, list):
        for item in rounds:
            if not isinstance(item, dict):
                continue
            text = " ".join(
                str(item.get(key, "")) for key in ("answer_summary", "transcript_summary", "follow_up_question")
            )
            if item.get("question_id") != current_question_id and text.strip():
                return True
            if any(marker in text for marker in forbidden_markers):
                return True
    return False
