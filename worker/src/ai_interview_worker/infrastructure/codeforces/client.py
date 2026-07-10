"""Codeforces HTTP Client 适配器。"""

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Any

import httpx

from ai_interview_worker.application.algorithm_test.codeforces_port import (
    CodeforcesFailure,
    CodeforcesProblem,
    CodeforcesProblemKey,
    CodeforcesProblemset,
    CodeforcesResult,
    CodeforcesSubmission,
    CodeforcesUser,
)

CF_HANDLE_NOT_FOUND = "CF_HANDLE_NOT_FOUND"
CF_API_UNAVAILABLE = "CF_API_UNAVAILABLE"


@dataclass(frozen=True)
class CodeforcesClientConfig:
    """Codeforces Client 配置。"""

    timeout_seconds: float
    max_submissions: int
    base_url: str = "https://codeforces.com/api"
    user_agent: str = "ai-interview-worker/1.0"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_submissions <= 0:
            raise ValueError("max_submissions must be positive")
        if not self.base_url.strip():
            raise ValueError("base_url is required")


class SharedRateLimiter:
    """同步共享限流器。"""

    def __init__(
        self,
        *,
        min_interval_seconds: float,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must not be negative")
        self._min_interval_seconds = min_interval_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._last_acquired_at: float | None = None

    @classmethod
    def disabled(cls) -> "SharedRateLimiter":
        """返回测试用禁用限流器。"""
        return cls(min_interval_seconds=0)

    def acquire(self) -> None:
        """在下一次外部请求前执行限流。"""
        now = self._clock()
        if self._last_acquired_at is not None:
            elapsed = now - self._last_acquired_at
            wait_seconds = self._min_interval_seconds - elapsed
            if wait_seconds > 0:
                self._sleeper(wait_seconds)
                now = self._clock()
        self._last_acquired_at = now


class HttpxCodeforcesClient:
    """基于 httpx 的 Codeforces Client。"""

    def __init__(
        self,
        *,
        config: CodeforcesClientConfig,
        transport: httpx.BaseTransport | None = None,
        rate_limiter: SharedRateLimiter | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._rate_limiter = rate_limiter or SharedRateLimiter(min_interval_seconds=1.0)

    def get_user_info(self, handle: str) -> CodeforcesResult:
        """查询账号信息。"""
        payload = self._get("/user.info", {"handles": handle})
        if not payload.success:
            return payload
        result = _payload_result(payload.data)
        if not isinstance(result, list) or not result:
            return _handle_not_found()
        first = result[0]
        if not isinstance(first, dict):
            return _api_unavailable()
        parsed_handle = first.get("handle")
        if not isinstance(parsed_handle, str) or not parsed_handle:
            return _api_unavailable()
        rating = first.get("rating")
        return CodeforcesResult(
            success=True,
            data=CodeforcesUser(handle=parsed_handle, rating=rating if isinstance(rating, int) else None),
        )

    def get_problemset(self, tags: tuple[str, ...] = ()) -> CodeforcesResult:
        """查询题库和 solvedCount。"""
        params: dict[str, str] = {}
        if tags:
            params["tags"] = ";".join(tags)
        payload = self._get("/problemset.problems", params)
        if not payload.success:
            return payload
        result = _payload_result(payload.data)
        if not isinstance(result, dict):
            return _api_unavailable()
        problems = tuple(_parse_problem(item) for item in result.get("problems", []) if isinstance(item, dict))
        statistics: dict[CodeforcesProblemKey, int] = {}
        for item in result.get("problemStatistics", []):
            if not isinstance(item, dict):
                continue
            key = _problem_key(item)
            solved_count = item.get("solvedCount")
            if key is not None and isinstance(solved_count, int):
                statistics[key] = solved_count
        return CodeforcesResult(success=True, data=CodeforcesProblemset(problems=problems, statistics=statistics))

    def get_user_status(self, handle: str, count: int) -> CodeforcesResult:
        """查询用户近期提交。"""
        if count <= 0 or count > self._config.max_submissions:
            return _api_unavailable()
        payload = self._get("/user.status", {"handle": handle, "from": "1", "count": str(count)})
        if not payload.success:
            return payload
        result = _payload_result(payload.data)
        if not isinstance(result, list):
            return _api_unavailable()
        submissions = tuple(_parse_submission(item) for item in result if isinstance(item, dict))
        return CodeforcesResult(success=True, data=submissions)

    def _get(self, path: str, params: dict[str, str]) -> CodeforcesResult:
        try:
            self._rate_limiter.acquire()
            with httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
                transport=self._transport,
                headers={"User-Agent": self._config.user_agent},
            ) as client:
                response = client.get(path, params=params)
            if response.status_code in (429,) or 500 <= response.status_code < 600:
                return _api_unavailable()
            if response.status_code != 200:
                return _api_unavailable()
            payload = response.json()
        except (httpx.TimeoutException, httpx.TransportError, ValueError):
            return _api_unavailable()

        if not isinstance(payload, dict):
            return _api_unavailable()
        if payload.get("status") == "OK":
            return CodeforcesResult(success=True, data=payload)
        comment = payload.get("comment")
        if isinstance(comment, str) and "not found" in comment.lower():
            return _handle_not_found()
        return _api_unavailable()


def _payload_result(payload: object | None) -> object | None:
    if isinstance(payload, dict):
        return payload.get("result")
    return None


def _problem_key(item: dict[str, Any]) -> CodeforcesProblemKey | None:
    contest_id = item.get("contestId")
    problem_index = item.get("index")
    if isinstance(contest_id, int) and isinstance(problem_index, str) and problem_index:
        return CodeforcesProblemKey(contest_id=contest_id, problem_index=problem_index)
    return None


def _parse_problem(item: dict[str, Any]) -> CodeforcesProblem:
    key = _problem_key(item) or CodeforcesProblemKey(contest_id=-1, problem_index="")
    name = item.get("name")
    rating = item.get("rating")
    tags = item.get("tags")
    problem_type = item.get("type")
    return CodeforcesProblem(
        key=key,
        name=name if isinstance(name, str) else "",
        rating=rating if isinstance(rating, int) else None,
        tags=tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else (),
        problem_type=problem_type if isinstance(problem_type, str) else "PROGRAMMING",
    )


def _parse_submission(item: dict[str, Any]) -> CodeforcesSubmission:
    problem = item.get("problem")
    key = _problem_key(problem) if isinstance(problem, dict) else None
    submission_id = item.get("id")
    created_at_seconds = item.get("creationTimeSeconds")
    verdict = item.get("verdict")
    return CodeforcesSubmission(
        submission_id=submission_id if isinstance(submission_id, int) else 0,
        created_at_seconds=created_at_seconds if isinstance(created_at_seconds, int) else 0,
        problem_key=key or CodeforcesProblemKey(contest_id=-1, problem_index=""),
        verdict=verdict if isinstance(verdict, str) else None,
    )


def _handle_not_found() -> CodeforcesResult:
    return CodeforcesResult(
        success=False,
        failure=CodeforcesFailure(
            failure_code=CF_HANDLE_NOT_FOUND,
            user_message="Codeforces 账号不存在，请确认后重新提交",
            retryable=False,
        ),
    )


def _api_unavailable() -> CodeforcesResult:
    return CodeforcesResult(
        success=False,
        failure=CodeforcesFailure(
            failure_code=CF_API_UNAVAILABLE,
            user_message="Codeforces 暂时不可用，请稍后重试",
            retryable=True,
        ),
    )
