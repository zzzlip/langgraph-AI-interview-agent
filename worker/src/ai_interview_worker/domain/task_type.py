"""Worker 支持的任务类型。"""

from enum import Enum


class TaskType(str, Enum):
    """平台可下发给 Python Worker 的任务类型。"""

    RESUME_EVALUATION = "resume_evaluation"
    RESUME_OPTIMIZATION = "resume_optimization"
    ALGORITHM_TEST = "algorithm_test"
    MOCK_INTERVIEW = "mock_interview"

    @classmethod
    def from_value(cls, value: str) -> "TaskType":
        """将字符串任务类型转换为枚举。"""
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unsupported task_type: {value}") from exc


COMMAND_EVENT_TYPES: dict[TaskType, str] = {
    TaskType.RESUME_EVALUATION: "resume_evaluation.command.created",
    TaskType.RESUME_OPTIMIZATION: "resume_optimization.command.created",
    TaskType.ALGORITHM_TEST: "algorithm_test.command.created",
    TaskType.MOCK_INTERVIEW: "mock_interview.command.created",
}
