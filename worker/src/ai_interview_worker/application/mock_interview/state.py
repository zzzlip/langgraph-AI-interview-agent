"""模拟面试 LangGraph state 定义。"""

from typing import Any, TypedDict


class MockInterviewState(TypedDict, total=False):
    """可序列化的模拟面试工作流状态。"""

    action: str
    task_id: str
    attempt: int
    current_question_id: str
    current_round_id: str
    current_question_title: str
    transcript: str
    transcript_summary: str
    context_snapshot: dict[str, Any]
    safety_blocked: bool
    safety_reason: str
    task_memory: dict[str, Any]
    input_summary: dict[str, Any]
    follow_up_question: str
    completion_reason: str
    route: str
    graph_path: list[str]
