"""模拟面试 LangGraph workflow。"""

from typing import Any, cast

from langgraph.graph import END, StateGraph

from ai_interview_worker.application.mock_interview.context_builder import build_follow_up_context
from ai_interview_worker.application.mock_interview.state import MockInterviewState


def build_mock_interview_workflow() -> Any:
    """构建模拟面试 StateGraph，只负责真实上下文裁剪。"""
    graph = StateGraph(MockInterviewState)
    graph.add_node("build_context", _build_context)
    graph.set_entry_point("build_context")
    graph.add_edge("build_context", END)
    return graph.compile()


def run_process_answer_workflow(input_summary: dict[str, Any], transcript: str) -> dict[str, Any]:
    """执行回答处理 workflow。"""
    workflow = build_mock_interview_workflow()
    return dict(
        workflow.invoke(
            {
                "action": "process_answer",
                "task_id": input_summary.get("task_id", ""),
                "attempt": int(input_summary.get("attempt", 1)),
                "transcript": transcript,
                "input_summary": input_summary,
            }
        )
    )


def _build_context(state: MockInterviewState) -> MockInterviewState:
    context = build_follow_up_context(state["input_summary"], state["transcript"])
    max_follow_up_rounds = int(state["input_summary"].get("max_follow_up_rounds", 2))
    return cast(
        MockInterviewState,
        {
            **context,
            "transcript": state["transcript"],
            "input_summary": state["input_summary"],
            "max_follow_up_rounds": max_follow_up_rounds,
            "graph_path": ["transcribe_answer", "build_follow_up_context"],
        },
    )
