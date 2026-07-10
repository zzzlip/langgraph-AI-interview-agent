"""算法测试业务 Handler。"""

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext, AbilityHandler
from ai_interview_worker.application.algorithm_test.codeforces_port import (
    CodeforcesClientPort,
    CodeforcesFailure,
    CodeforcesProblemKey,
    CodeforcesSubmission,
)
from ai_interview_worker.application.algorithm_test.problem_selector import (
    AlgorithmProblemSelectionService,
    SelectedAlgorithmProblem,
)
from ai_interview_worker.application.feedback_service import FeedbackService
from ai_interview_worker.application.llm_config import llm_request_kwargs
from ai_interview_worker.application.port.file_ports import build_artifact_object_key
from ai_interview_worker.domain.artifact import OutputArtifact
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.result import WorkerSuccessResult


class AlgorithmTestHandler(AbilityHandler):
    """算法测试题单生成 Handler。"""

    def __init__(self, *, codeforces_client: CodeforcesClientPort, max_submissions: int = 1000) -> None:
        self._codeforces_client = codeforces_client
        self._max_submissions = max_submissions

    def handle(self, ctx: AbilityContext) -> WorkerSuccessResult:
        """执行算法测试命令。"""
        action = _input_summary_value(ctx, "action")
        if action == "generate_problem_set":
            return self._generate_problem_set(ctx)
        if action == "check_submission":
            return self._check_submission(ctx)
        raise _failure(
            WorkerFailureCode.INPUT_ERROR,
            "不支持的算法测试动作",
            failed_stage="algorithm_test_command",
            retryable=False,
        )

    def _generate_problem_set(self, ctx: AbilityContext) -> WorkerSuccessResult:
        target_job = _target_job(ctx)
        handle = _required_summary_text(ctx, "codeforces_handle")
        problem_count = _summary_int(ctx, "problem_count", 5)
        pass_threshold = _summary_int(ctx, "pass_threshold", 2)
        if problem_count != 5 or pass_threshold <= 0 or pass_threshold > problem_count:
            raise _failure(
                WorkerFailureCode.INPUT_ERROR,
                "算法测试命令参数不合法",
                failed_stage="algorithm_test_command",
                retryable=False,
            )

        ctx.progress_reporter(stage="cf_account_validation", progress=10, message="正在校验 Codeforces 账号")
        self._validate_handle(handle)

        ctx.progress_reporter(stage="problem_generation", progress=45, message="正在生成算法测试题单")
        if ctx.ai_gateway is None:
            raise _failure(
                WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
                "岗位能力分析服务暂时不可用，请稍后重试",
                failed_stage="problem_generation",
                retryable=True,
            )
        selection = AlgorithmProblemSelectionService(
            ai_gateway=ctx.ai_gateway,
            codeforces_client=self._codeforces_client,
            max_submissions=self._max_submissions,
            llm_request_kwargs=llm_request_kwargs(ctx.command),
        ).generate_problem_set(target_job["title"], target_job["description"], handle)
        if not selection.success or selection.failure is not None or selection.problems is None:
            raise ValueError(selection.failure)

        result_summary = _problem_set_summary(
            handle=handle,
            problem_count=problem_count,
            pass_threshold=pass_threshold,
            selection_tags=selection.capability_plan.tags if selection.capability_plan else (),
            rating_min=selection.capability_plan.rating_min if selection.capability_plan else 800,
            rating_max=selection.capability_plan.rating_max if selection.capability_plan else 1700,
            problems=selection.problems,
        )
        output_files = _write_optional_json_artifact(
            ctx,
            result_summary,
            file_purpose="algorithm_test_problem_set",
            display_name="算法测试题单.json",
            failed_stage="problem_generation",
        )
        result = WorkerSuccessResult(result_summary=result_summary, output_files=output_files)
        ctx.progress_reporter(stage="waiting_user_input", progress=95, message="题单已生成，正在保存等待答题状态")
        _publish_problem_set_feedback(ctx, result)
        return result

    def _check_submission(self, ctx: AbilityContext) -> WorkerSuccessResult:
        handle = _required_summary_text(ctx, "codeforces_handle")
        problem_count = _summary_int(ctx, "problem_count", 5)
        pass_threshold = _summary_int(ctx, "pass_threshold", 2)
        problem_set = _check_problem_set(ctx, problem_count)

        ctx.progress_reporter(stage="codeforces_check", progress=50, message="正在检查 Codeforces 提交记录")
        status_result = self._codeforces_client.get_user_status(handle, count=self._max_submissions)
        if not status_result.success or not isinstance(status_result.data, Sequence):
            raise ValueError(_codeforces_failure_to_worker_failure(status_result.failure, "codeforces_check"))
        submissions = [item for item in status_result.data if isinstance(item, CodeforcesSubmission)]
        result_summary = _check_result_summary(
            handle=handle,
            problem_set=problem_set,
            submissions=submissions,
            pass_threshold=pass_threshold,
        )
        output_files = _write_optional_json_artifact(
            ctx,
            result_summary,
            file_purpose="algorithm_test_result",
            display_name="算法测试结果.json",
            failed_stage="codeforces_check",
        )
        ctx.progress_reporter(stage="completed", progress=100, message="算法测试提交检查已完成")
        return WorkerSuccessResult(result_summary=result_summary, output_files=output_files)

    def _validate_handle(self, handle: str) -> None:
        result = self._codeforces_client.get_user_info(handle)
        if result.success:
            return
        failure = result.failure
        raise ValueError(_codeforces_failure_to_worker_failure(failure, "cf_account_validation"))


def _publish_problem_set_feedback(ctx: AbilityContext, result: WorkerSuccessResult) -> None:
    if ctx.feedback_publisher is None:
        return
    service = FeedbackService(
        event_id_supplier=ctx.event_id_supplier or (lambda: "algorithm-problem-set-feedback"),
        clock=ctx.clock or _utc_now,
    )
    ctx.feedback_publisher(service.algorithm_problem_set_event(ctx.command, result))


def _write_optional_json_artifact(
    ctx: AbilityContext,
    result_summary: dict[str, Any],
    *,
    file_purpose: str,
    display_name: str,
    failed_stage: str,
) -> list[OutputArtifact]:
    if ctx.artifact_writer is None:
        return []
    data = json.dumps(result_summary, ensure_ascii=False, sort_keys=True).encode("utf-8")
    object_key = build_artifact_object_key(
        ctx.command.task_id,
        ctx.command.attempt,
        file_purpose,
        display_name,
    )
    write_result = ctx.artifact_writer.put_object(
        object_key=object_key,
        data=data,
        mime_type="application/json",
        display_name=display_name,
        clock=ctx.clock or _utc_now,
    )
    if not write_result.success or write_result.artifact is None:
        raise _failure(
            WorkerFailureCode.ARTIFACT_WRITE_ERROR,
            "算法测试 JSON 产物写入失败，请稍后重试",
            failed_stage=failed_stage,
            retryable=True,
        )
    artifact = write_result.artifact
    if artifact.file_type != "json" or artifact.file_purpose != file_purpose:
        raise _failure(
            WorkerFailureCode.ARTIFACT_WRITE_ERROR,
            "算法测试 JSON 产物元数据非法",
            failed_stage=failed_stage,
            retryable=True,
        )
    return [artifact]


def _problem_set_summary(
    *,
    handle: str,
    problem_count: int,
    pass_threshold: int,
    selection_tags: tuple[str, ...],
    rating_min: int,
    rating_max: int,
    problems: tuple[SelectedAlgorithmProblem, ...],
) -> dict[str, Any]:
    return {
        "result_type": "algorithm_test",
        "schema_version": 1,
        "phase": "problem_set",
        "handle": handle,
        "handle_valid": True,
        "problem_count": problem_count,
        "pass_threshold": pass_threshold,
        "selection": {
            "tags": list(selection_tags),
            "rating_min": rating_min,
            "rating_max": rating_max,
            "strategy_version": "cf-v1",
        },
        "problems": [_problem_to_payload(index, problem) for index, problem in enumerate(problems, start=1)],
    }


def _problem_to_payload(index: int, problem: SelectedAlgorithmProblem) -> dict[str, Any]:
    return {
        "id": index,
        "contest_id": problem.key.contest_id,
        "problem_index": problem.key.problem_index,
        "title": problem.name,
        "link": problem.link,
        "difficulty": problem.difficulty,
        "rating": problem.rating,
        "tags": list(problem.tags),
        "solved_count": problem.solved_count,
    }


def _check_problem_set(ctx: AbilityContext, expected_count: int) -> list[dict[str, Any]]:
    raw_problems = _input_summary_value(ctx, "problems")
    if not isinstance(raw_problems, list) or len(raw_problems) != expected_count:
        raise _problem_set_failure()
    problems: list[dict[str, Any]] = []
    seen: set[CodeforcesProblemKey] = set()
    for item in raw_problems:
        if not isinstance(item, dict):
            raise _problem_set_failure()
        contest_id = item.get("contest_id")
        problem_index = item.get("problem_index")
        title = item.get("title")
        if not isinstance(contest_id, int) or contest_id <= 0:
            raise _problem_set_failure()
        if not isinstance(problem_index, str) or not problem_index.strip():
            raise _problem_set_failure()
        if not isinstance(title, str) or not title.strip():
            raise _problem_set_failure()
        key = CodeforcesProblemKey(contest_id, problem_index.strip())
        if key in seen:
            raise _problem_set_failure()
        seen.add(key)
        problems.append(
            {
                **item,
                "contest_id": contest_id,
                "problem_index": problem_index.strip(),
                "title": title.strip(),
            }
        )
    return problems


def _check_result_summary(
    *,
    handle: str,
    problem_set: list[dict[str, Any]],
    submissions: list[CodeforcesSubmission],
    pass_threshold: int,
) -> dict[str, Any]:
    ok_records: dict[CodeforcesProblemKey, CodeforcesSubmission] = {}
    problem_keys = {
        CodeforcesProblemKey(problem["contest_id"], problem["problem_index"])
        for problem in problem_set
    }
    for submission in submissions:
        if submission.problem_key not in problem_keys or submission.verdict != "OK":
            continue
        ok_records.setdefault(submission.problem_key, submission)

    problem_results = [
        _problem_check_payload(
            index,
            problem,
            ok_records.get(CodeforcesProblemKey(problem["contest_id"], problem["problem_index"])),
        )
        for index, problem in enumerate(problem_set, start=1)
    ]
    passed_count = sum(1 for item in problem_results if item["passed"] is True)
    passed = passed_count >= pass_threshold
    return {
        "result_type": "algorithm_test",
        "schema_version": 1,
        "phase": "check_result",
        "handle": handle,
        "passed_count": passed_count,
        "total_count": len(problem_set),
        "passed": passed,
        "verdict": "整体表现良好，可进入面试准备" if passed else "建议继续完成题单后再进入面试准备",
        "problem_results": problem_results,
    }


def _problem_check_payload(
    index: int,
    problem: dict[str, Any],
    ok_submission: CodeforcesSubmission | None,
) -> dict[str, Any]:
    payload = {
        "id": index,
        "title": problem["title"],
        "contest_id": problem["contest_id"],
        "problem_index": problem["problem_index"],
        "passed": ok_submission is not None,
        "verdict": "OK" if ok_submission is not None else "WRONG_ANSWER",
    }
    if ok_submission is not None:
        payload["submission_id"] = ok_submission.submission_id
        payload["submitted_at"] = datetime.fromtimestamp(
            ok_submission.created_at_seconds,
            tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")
    return payload


def _target_job(ctx: AbilityContext) -> dict[str, str]:
    value = _input_summary_value(ctx, "target_job")
    if not isinstance(value, dict):
        raise _command_failure()
    title = value.get("title")
    description = value.get("description")
    if not isinstance(title, str) or not title.strip() or not isinstance(description, str) or not description.strip():
        raise _command_failure()
    return {"title": title.strip(), "description": description.strip()}


def _required_summary_text(ctx: AbilityContext, key: str) -> str:
    value = _input_summary_value(ctx, key)
    if not isinstance(value, str) or not value.strip():
        raise _command_failure()
    return value.strip()


def _summary_int(ctx: AbilityContext, key: str, default: int) -> int:
    value = _input_summary_value(ctx, key)
    return value if isinstance(value, int) else default


def _input_summary_value(ctx: AbilityContext, key: str) -> Any:
    return (ctx.command.input_summary or {}).get(key)


def _command_failure() -> ValueError:
    return _failure(
        WorkerFailureCode.INPUT_ERROR,
        "算法测试命令参数不合法",
        failed_stage="algorithm_test_command",
        retryable=False,
    )


def _problem_set_failure() -> ValueError:
    return _failure(
        WorkerFailureCode.INPUT_ERROR,
        "题单数据不完整，无法检查提交",
        failed_stage="codeforces_check",
        retryable=False,
    )


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
        user_message=(failure.user_message if failure is not None else "Codeforces 账号不可用，请确认后重新提交"),
        failed_stage=failed_stage,
        retryable=False,
    )


def _failure(
    failure_code: WorkerFailureCode,
    user_message: str,
    *,
    failed_stage: str,
    retryable: bool,
) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=failure_code,
            user_message=user_message,
            failed_stage=failed_stage,
            retryable=retryable,
        )
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
