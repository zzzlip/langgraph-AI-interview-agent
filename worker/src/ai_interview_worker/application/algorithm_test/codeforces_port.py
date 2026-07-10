"""Codeforces Client Port 与数据模型。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CodeforcesProblemKey:
    """Codeforces 稳定判题键。"""

    contest_id: int
    problem_index: str


@dataclass(frozen=True)
class CodeforcesUser:
    """Codeforces 账号信息。"""

    handle: str
    rating: int | None = None


@dataclass(frozen=True)
class CodeforcesProblem:
    """Codeforces 题目信息。"""

    key: CodeforcesProblemKey
    name: str
    rating: int | None
    tags: tuple[str, ...]
    problem_type: str = "PROGRAMMING"


@dataclass(frozen=True)
class CodeforcesProblemset:
    """Codeforces 题库快照。"""

    problems: tuple[CodeforcesProblem, ...]
    statistics: dict[CodeforcesProblemKey, int]


@dataclass(frozen=True)
class CodeforcesSubmission:
    """Codeforces 提交记录。"""

    submission_id: int
    created_at_seconds: int
    problem_key: CodeforcesProblemKey
    verdict: str | None


@dataclass(frozen=True)
class CodeforcesFailure:
    """Codeforces 调用失败。"""

    failure_code: str
    user_message: str
    retryable: bool


@dataclass(frozen=True)
class CodeforcesResult:
    """Codeforces 调用结果。"""

    success: bool
    data: object | None = None
    failure: CodeforcesFailure | None = None


class CodeforcesClientPort(Protocol):
    """Codeforces Client 唯一应用层入口。"""

    def get_user_info(self, handle: str) -> CodeforcesResult:
        """查询并校验账号。"""
        ...

    def get_problemset(self, tags: tuple[str, ...] = ()) -> CodeforcesResult:
        """查询题库。"""
        ...

    def get_user_status(self, handle: str, count: int) -> CodeforcesResult:
        """查询用户近期提交。"""
        ...
