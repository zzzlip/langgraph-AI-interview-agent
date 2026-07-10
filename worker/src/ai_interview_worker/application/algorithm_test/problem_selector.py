"""算法测试岗位能力分析与题目筛选。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from ai_interview_worker.application.algorithm_test.codeforces_port import (
    CodeforcesClientPort,
    CodeforcesFailure,
    CodeforcesProblem,
    CodeforcesProblemKey,
    CodeforcesProblemset,
    CodeforcesSubmission,
)
from ai_interview_worker.application.llm_config import chat_json
from ai_interview_worker.application.port.ai_gateway_port import AiChatRequest, AiGatewayPort
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode

RATING_MIN = 800
RATING_MAX = 1700
PROBLEM_COUNT = 5
DEFAULT_TAGS = ("implementation", "data structures", "greedy")
DIFFICULTY_TARGET_RATING = {"easy": 800, "medium": 1200, "hard": 1400}
DEFAULT_DIFFICULTY_PLAN = (("easy", 2), ("medium", 2), ("hard", 1))
ALLOWED_CODEFORCES_TAGS = {
    "2-sat",
    "binary search",
    "bitmasks",
    "brute force",
    "chinese remainder theorem",
    "combinatorics",
    "constructive algorithms",
    "data structures",
    "dfs and similar",
    "divide and conquer",
    "dp",
    "dsu",
    "expression parsing",
    "fft",
    "flows",
    "games",
    "geometry",
    "graph matchings",
    "graphs",
    "greedy",
    "hashing",
    "implementation",
    "interactive",
    "math",
    "matrices",
    "meet-in-the-middle",
    "number theory",
    "probabilities",
    "schedules",
    "shortest paths",
    "sortings",
    "string suffix structures",
    "strings",
    "ternary search",
    "trees",
    "two pointers",
}


@dataclass(frozen=True)
class DifficultyBucket:
    """岗位题单难度桶。"""

    difficulty: str
    count: int


@dataclass(frozen=True)
class JobCapabilityPlan:
    """岗位算法能力分析结果。"""

    tags: tuple[str, ...]
    rating_min: int
    rating_max: int
    difficulty_plan: tuple[DifficultyBucket, ...]


@dataclass(frozen=True)
class SelectedAlgorithmProblem:
    """最终推荐题目。"""

    key: CodeforcesProblemKey
    name: str
    rating: int
    tags: tuple[str, ...]
    solved_count: int
    link: str
    difficulty: str


@dataclass(frozen=True)
class AlgorithmProblemSelectionResult:
    """题目筛选结果。"""

    success: bool
    problems: tuple[SelectedAlgorithmProblem, ...] | None = None
    failure: WorkerFailure | None = None
    capability_plan: JobCapabilityPlan | None = None


class AlgorithmProblemSelectionService:
    """根据岗位和 Codeforces 真实题库生成稳定题单。"""

    def __init__(
        self,
        *,
        ai_gateway: AiGatewayPort,
        codeforces_client: CodeforcesClientPort,
        max_submissions: int = 1000,
        llm_request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._ai_gateway = ai_gateway
        self._codeforces_client = codeforces_client
        self._max_submissions = max_submissions
        self._llm_request_kwargs = llm_request_kwargs or {}

    def generate_problem_set(
        self,
        target_job_title: str,
        target_job_description: str,
        codeforces_handle: str,
    ) -> AlgorithmProblemSelectionResult:
        """生成 5 道真实 Codeforces 题单。"""
        capability_result = self._analyze_job_capability(target_job_title, target_job_description)
        if capability_result.failure is not None or capability_result.capability_plan is None:
            return capability_result
        plan = capability_result.capability_plan

        problemset_result = self._codeforces_client.get_problemset(tags=plan.tags)
        if not problemset_result.success or not isinstance(problemset_result.data, CodeforcesProblemset):
            return _failure_result(
                _codeforces_failure_to_worker_failure(problemset_result.failure, "problem_generation")
            )
        problemset = problemset_result.data

        status_result = self._codeforces_client.get_user_status(codeforces_handle, count=self._max_submissions)
        if not status_result.success or not isinstance(status_result.data, Sequence):
            return _failure_result(_codeforces_failure_to_worker_failure(status_result.failure, "problem_generation"))
        submissions = cast(Sequence[CodeforcesSubmission], status_result.data)
        solved_keys = {submission.problem_key for submission in submissions if submission.verdict == "OK"}

        selected = select_problems(problemset, plan, solved_keys)
        if len(selected) < PROBLEM_COUNT:
            return _failure_result(
                WorkerFailure(
                    failure_code=WorkerFailureCode.INPUT_ERROR,
                    user_message="符合条件的算法题不足，请调整岗位信息后重新生成题单",
                    failed_stage="problem_generation",
                    retryable=False,
                )
            )
        return AlgorithmProblemSelectionResult(
            success=True,
            problems=tuple(selected),
            capability_plan=plan,
        )

    def _analyze_job_capability(
        self,
        target_job_title: str,
        target_job_description: str,
    ) -> AlgorithmProblemSelectionResult:
        request = AiChatRequest(
            messages=[
                {
                    "role": "system",
                    "content": "根据岗位信息输出算法题筛选 JSON，不要输出解释文本。",
                },
                {
                    "role": "user",
                    "content": (
                        f"岗位标题：{target_job_title}\n"
                        f"岗位描述：{target_job_description}\n"
                        f"Codeforces 标签白名单：{', '.join(sorted(ALLOWED_CODEFORCES_TAGS))}\n"
                        "rating 范围：800-1700，题目总数固定 5。"
                    ),
                },
            ],
            temperature=0.1,
            **self._llm_request_kwargs,
        )
        result = self._ai_gateway.chat_completion(request)
        if not result.success:
            return _failure_result(
                WorkerFailure(
                    failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
                    user_message="岗位能力分析服务暂时不可用，请稍后重试",
                    failed_stage="problem_generation",
                    retryable=True,
                )
            )
        try:
            raw_plan = chat_json(result.data)
        except ValueError:
            return _failure_result(
                WorkerFailure(
                    failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
                    user_message="岗位能力分析结果格式异常，请稍后重试",
                    failed_stage="problem_generation",
                    retryable=True,
                )
            )
        return AlgorithmProblemSelectionResult(
            success=True,
            capability_plan=normalize_capability_plan(raw_plan),
        )


def normalize_capability_plan(raw_plan: dict[str, Any]) -> JobCapabilityPlan:
    """归一化 AI Gateway 输出，禁止把异常输出直接用于筛题。"""
    tags = _normalize_tags(raw_plan.get("tags"))
    rating_min = _clamp_rating(raw_plan.get("rating_min"), RATING_MIN)
    rating_max = _clamp_rating(raw_plan.get("rating_max"), RATING_MAX)
    if rating_min > rating_max:
        rating_min, rating_max = rating_max, rating_min

    difficulty_plan = _normalize_difficulty_plan(raw_plan.get("difficulty_plan"))
    return JobCapabilityPlan(
        tags=tags,
        rating_min=rating_min,
        rating_max=rating_max,
        difficulty_plan=difficulty_plan,
    )


def select_problems(
    problemset: CodeforcesProblemset,
    plan: JobCapabilityPlan,
    solved_keys: set[CodeforcesProblemKey],
) -> tuple[SelectedAlgorithmProblem, ...]:
    """按 plan 从题库中筛选并稳定排序。"""
    candidates = _deduplicate_candidates(problemset, plan, solved_keys)
    used_keys: set[CodeforcesProblemKey] = set()
    selected: list[SelectedAlgorithmProblem] = []

    for bucket in plan.difficulty_plan:
        bucket_candidates = [
            item for item in candidates if item.difficulty == bucket.difficulty and item.key not in used_keys
        ]
        for item in bucket_candidates[: bucket.count]:
            selected.append(item)
            used_keys.add(item.key)

    if len(selected) < PROBLEM_COUNT:
        remaining = [item for item in candidates if item.key not in used_keys]
        needed = PROBLEM_COUNT - len(selected)
        selected.extend(remaining[:needed])

    return tuple(
        sorted(
            selected[:PROBLEM_COUNT],
            key=lambda item: (
                _difficulty_order(item.difficulty),
                item.rating,
                item.key.contest_id,
                item.key.problem_index,
            ),
        )
    )


def _deduplicate_candidates(
    problemset: CodeforcesProblemset,
    plan: JobCapabilityPlan,
    solved_keys: set[CodeforcesProblemKey],
) -> list[SelectedAlgorithmProblem]:
    seen: set[CodeforcesProblemKey] = set()
    candidates: list[SelectedAlgorithmProblem] = []
    for problem in problemset.problems:
        if problem.key in seen or problem.key in solved_keys:
            continue
        seen.add(problem.key)
        if not _is_valid_problem(problem, plan):
            continue
        rating = cast(int, problem.rating)
        difficulty = _difficulty_for_rating(rating)
        candidates.append(
            SelectedAlgorithmProblem(
                key=problem.key,
                name=problem.name,
                rating=rating,
                tags=problem.tags,
                solved_count=problemset.statistics.get(problem.key, 0),
                link=f"https://codeforces.com/problemset/problem/{problem.key.contest_id}/{problem.key.problem_index}",
                difficulty=difficulty,
            )
        )
    return sorted(candidates, key=lambda item: _candidate_sort_key(item, plan))


def _candidate_sort_key(item: SelectedAlgorithmProblem, plan: JobCapabilityPlan) -> tuple[int, int, int, int, str]:
    matched_tag_count = len(set(item.tags).intersection(plan.tags))
    target_rating = DIFFICULTY_TARGET_RATING[item.difficulty]
    return (
        -matched_tag_count,
        abs(item.rating - target_rating),
        -item.solved_count,
        -item.key.contest_id,
        item.key.problem_index,
    )


def _is_valid_problem(problem: CodeforcesProblem, plan: JobCapabilityPlan) -> bool:
    if problem.problem_type != "PROGRAMMING":
        return False
    if problem.key.contest_id <= 0 or not problem.key.problem_index.strip():
        return False
    if problem.rating is None or problem.rating < RATING_MIN or problem.rating > RATING_MAX:
        return False
    if problem.rating < plan.rating_min or problem.rating > plan.rating_max:
        return False
    return bool(set(problem.tags).intersection(plan.tags))


def _normalize_tags(raw_tags: Any) -> tuple[str, ...]:
    if not isinstance(raw_tags, list | tuple):
        return DEFAULT_TAGS
    normalized: list[str] = []
    for raw_tag in raw_tags:
        if not isinstance(raw_tag, str):
            continue
        tag = raw_tag.strip()
        if tag in ALLOWED_CODEFORCES_TAGS and tag not in normalized:
            normalized.append(tag)
    return tuple(normalized) or DEFAULT_TAGS


def _clamp_rating(raw_rating: Any, default: int) -> int:
    if not isinstance(raw_rating, int):
        return default
    return min(max(raw_rating, RATING_MIN), RATING_MAX)


def _normalize_difficulty_plan(raw_plan: Any) -> tuple[DifficultyBucket, ...]:
    if not isinstance(raw_plan, list):
        return tuple(DifficultyBucket(name, count) for name, count in DEFAULT_DIFFICULTY_PLAN)
    buckets: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        difficulty = item.get("difficulty")
        count = item.get("count")
        if difficulty in buckets and isinstance(count, int) and count > 0:
            buckets[difficulty] += count
    if sum(buckets.values()) != PROBLEM_COUNT:
        return tuple(DifficultyBucket(name, count) for name, count in DEFAULT_DIFFICULTY_PLAN)
    return tuple(DifficultyBucket(name, buckets[name]) for name in ("easy", "medium", "hard"))


def _difficulty_for_rating(rating: int) -> str:
    if rating <= 999:
        return "easy"
    if rating <= 1399:
        return "medium"
    return "hard"


def _difficulty_order(difficulty: str) -> int:
    return {"easy": 0, "medium": 1, "hard": 2}[difficulty]


def _codeforces_failure_to_worker_failure(
    failure: CodeforcesFailure | None,
    failed_stage: str,
) -> WorkerFailure:
    if failure is not None and failure.retryable:
        return WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
            user_message=failure.user_message,
            failed_stage=failed_stage,
            retryable=True,
        )
    return WorkerFailure(
        failure_code=WorkerFailureCode.INPUT_ERROR,
        user_message=(failure.user_message if failure is not None else "Codeforces 数据不可用，请稍后重试"),
        failed_stage=failed_stage,
        retryable=False,
    )


def _failure_result(failure: WorkerFailure) -> AlgorithmProblemSelectionResult:
    return AlgorithmProblemSelectionResult(success=False, failure=failure)
