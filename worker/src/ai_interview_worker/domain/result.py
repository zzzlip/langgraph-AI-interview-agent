from dataclasses import dataclass
from typing import Any

from ai_interview_worker.domain.artifact import OutputArtifact


@dataclass(frozen=True)
class WorkerSuccessResult:
    """Worker 成功执行后的结构化结果。"""

    result_summary: dict[str, Any]
    output_files: list[OutputArtifact] | None = None

    def __post_init__(self) -> None:
        if "schema_version" not in self.result_summary:
            raise ValueError("result_summary.schema_version is required")
        if self.output_files is None:
            object.__setattr__(self, "output_files", [])


@dataclass
class TaskResult:
    """兼容 spec01 dry-run 的旧结果类型。"""

    task_id: str
    status: str          # success / failed
    output_data: dict[str, Any] | None = None
