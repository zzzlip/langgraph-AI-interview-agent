"""简历评估业务 Handler。"""

import math
import re
from datetime import datetime, timezone
from typing import Any

from ai_interview_worker.application.ability_handler import AbilityContext, AbilityHandler
from ai_interview_worker.application.llm_config import chat_json, llm_request_kwargs
from ai_interview_worker.application.port.ai_gateway_port import AiChatRequest, AiEmbeddingRequest, AiSearchRequest
from ai_interview_worker.application.port.file_ports import build_artifact_object_key
from ai_interview_worker.application.resume_evaluation.artifacts import (
    render_evaluation_report,
    render_radar_chart,
)
from ai_interview_worker.application.resume_evaluation.schema import validate_resume_evaluation_command
from ai_interview_worker.application.resume_evaluation.scoring_config import (
    DIMENSION_ORDER,
    config_for_profile,
)
from ai_interview_worker.domain.artifact import OutputArtifact
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode
from ai_interview_worker.domain.result import WorkerSuccessResult
from ai_interview_worker.legacy_adapter.resume_evaluation import ParsedResume, parse_resume


class ResumeEvaluationHandler(AbilityHandler):
    """简历评估业务 Handler。"""

    def handle(self, ctx: AbilityContext) -> WorkerSuccessResult:
        business_payload = validate_resume_evaluation_command(ctx.command)
        resume_file = next(file for file in ctx.command.input_files or [] if file.file_purpose == "resume")
        ctx.progress_reporter(stage="resume_parsing", progress=10, message="正在解析简历")
        parsed_resume = self._parse_resume(ctx, resume_file.object_key, resume_file.file_type)
        options = business_payload.get("evaluation_options", {})
        profile = options.get("dimension_profile", "default_v1") if isinstance(options, dict) else "default_v1"
        scoring_config = config_for_profile(profile)
        ctx.progress_reporter(stage="ai_scoring", progress=45, message="正在进行五维评分")
        score = self._score_resume(ctx, parsed_resume, business_payload, scoring_config.weights)
        result_summary = {
            "schema_version": 1,
            "result_type": "resume_evaluation",
            "overall_score": score["overall_score"],
            "summary": score["summary"],
            "dimensions": score["dimensions"],
            "artifact_warnings": [
                {"artifact": "dimension_profile", "message": "未知评分配置，已使用默认权重"}
                for warning in scoring_config.warnings
                if warning == "unknown_dimension_profile"
            ],
        }
        output_files = self._generate_artifacts(ctx, result_summary)
        return WorkerSuccessResult(result_summary=result_summary, output_files=output_files)

    def _parse_resume(self, ctx: AbilityContext, object_key: str, file_type: str) -> ParsedResume:
        if ctx.minio_reader is None:
            raise _failure(
                WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
                "文件存储服务暂时不可用",
                retryable=True,
            )
        read_result = ctx.minio_reader.get_object(object_key)
        if not read_result.success or read_result.data is None:
            raise _failure(
                WorkerFailureCode.INPUT_ERROR,
                "简历文件无法读取，请重新上传后再试",
                retryable=False,
            )
        try:
            return parse_resume(read_result.data, file_type)
        except ValueError as exc:
            raise _failure(
                WorkerFailureCode.INPUT_ERROR,
                "简历文件无法解析，请确认文件内容清晰后重新上传",
                retryable=False,
            ) from exc

    def _score_resume(
        self,
        ctx: AbilityContext,
        parsed_resume: ParsedResume,
        business_payload: dict[str, Any],
        weights: dict[str, int],
    ) -> dict[str, Any]:
        if ctx.ai_gateway is None:
            raise _ai_failure("AI 评分服务暂时不可用，请稍后重试")
        target_job = business_payload["target_job"]
        search_context = _search_context(ctx, parsed_resume, target_job)
        keyword_context = _keyword_semantic_context(ctx, parsed_resume, target_job)
        dimensions: list[dict[str, Any]] = []
        overall_summary = ""
        for dimension_name in DIMENSION_ORDER:
            dimension, summary = self._score_dimension(
                ctx,
                dimension_name=dimension_name,
                parsed_resume=parsed_resume,
                target_job=target_job,
                search_context=search_context,
                keyword_context=keyword_context,
            )
            dimensions.append(dimension)
            if not overall_summary and summary:
                overall_summary = summary
        overall_score = round(sum(dim["score"] * weights[dim["name"]] for dim in dimensions) / 100)
        return {
            "overall_score": overall_score,
            "summary": overall_summary or _build_summary(dimensions),
            "dimensions": dimensions,
        }

    def _score_dimension(
        self,
        ctx: AbilityContext,
        *,
        dimension_name: str,
        parsed_resume: ParsedResume,
        target_job: dict[str, Any],
        search_context: str,
        keyword_context: str,
    ) -> tuple[dict[str, Any], str]:
        if ctx.ai_gateway is None:
            raise _ai_failure("AI 评分服务暂时不可用，请稍后重试")
        request = AiChatRequest(
            messages=_dimension_messages(
                dimension_name=dimension_name,
                parsed_resume=parsed_resume,
                target_job=target_job,
                search_context=search_context,
                keyword_context=keyword_context,
            ),
            temperature=0.2,
            **llm_request_kwargs(ctx.command),
        )
        result = ctx.ai_gateway.chat_completion(request)
        if not result.success:
            raise _ai_failure("AI 评分服务暂时不可用，请稍后重试")
        payload = chat_json(result.data)
        summary = payload.get("summary")
        return _validate_dimension_payload(payload, dimension_name), summary.strip() if isinstance(summary, str) else ""

    def _generate_artifacts(self, ctx: AbilityContext, result_summary: dict[str, Any]) -> list[OutputArtifact]:
        output_files: list[OutputArtifact] = []
        radar_chart_bytes = None
        ctx.progress_reporter(stage="radar_chart_generation", progress=75, message="正在生成雷达图")
        try:
            radar_chart_bytes = render_radar_chart(result_summary)
            artifact = self._write_artifact(
                ctx,
                data=radar_chart_bytes,
                file_purpose="radar_chart",
                file_type="png",
                mime_type="image/png",
                display_name="简历评估雷达图.png",
            )
            if artifact is not None:
                output_files.append(artifact)
        except Exception:
            result_summary["artifact_warnings"].append(
                {"artifact": "radar_chart", "message": "雷达图生成失败，评分结果仍可查看"}
            )

        ctx.progress_reporter(stage="report_generation", progress=90, message="正在生成评估报告")
        try:
            report_bytes = render_evaluation_report(result_summary, radar_chart_bytes)
            artifact = self._write_artifact(
                ctx,
                data=report_bytes,
                file_purpose="evaluation_report",
                file_type="docx",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                display_name="简历评估报告.docx",
            )
            if artifact is not None:
                output_files.append(artifact)
        except Exception:
            result_summary["artifact_warnings"].append(
                {"artifact": "evaluation_report", "message": "评估报告生成失败，评分结果仍可查看"}
            )
        return output_files

    def _write_artifact(
        self,
        ctx: AbilityContext,
        *,
        data: bytes,
        file_purpose: str,
        file_type: str,
        mime_type: str,
        display_name: str,
    ) -> OutputArtifact | None:
        if ctx.artifact_writer is None:
            return None
        object_key = build_artifact_object_key(ctx.command.task_id, ctx.command.attempt, file_purpose, display_name)
        result = ctx.artifact_writer.put_object(
            object_key=object_key,
            data=data,
            mime_type=mime_type,
            display_name=display_name,
            clock=ctx.clock or (lambda: datetime.now(timezone.utc)),
        )
        if not result.success or result.artifact is None:
            raise ValueError("评估产物写入失败")
        if result.artifact.file_type != file_type or result.artifact.file_purpose != file_purpose:
            raise ValueError("评估产物元数据非法")
        return result.artifact


def _failure(failure_code: WorkerFailureCode, user_message: str, *, retryable: bool) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=failure_code,
            user_message=user_message,
            failed_stage="resume_parsing",
            retryable=retryable,
        )
    )


def _ai_failure(user_message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.EXTERNAL_SERVICE_ERROR,
            user_message=user_message,
            failed_stage="ai_scoring",
            retryable=True,
        )
    )


def _validate_dimension_payload(data: dict[str, Any], dimension_name: str) -> dict[str, Any]:
    raw_dimensions = data.get("dimensions")
    if isinstance(raw_dimensions, list):
        names = [item.get("name") for item in raw_dimensions if isinstance(item, dict)]
        if len(names) != len(raw_dimensions) or tuple(names) != DIMENSION_ORDER:
            raise _ai_failure("AI 评分结果格式异常，请稍后重试")
        matched_dimension = next(
            (item for item in raw_dimensions if isinstance(item, dict) and item.get("name") == dimension_name),
            None,
        )
        if matched_dimension is None:
            raise _ai_failure("AI 评分结果格式异常，请稍后重试")
        data = matched_dimension
    name = data.get("name", dimension_name)
    score = data.get("score")
    analysis = data.get("analysis")
    if name != dimension_name:
        raise _ai_failure("AI 评分结果格式异常，请稍后重试")
    if not isinstance(score, int) or score < 0 or score > 100:
        raise _ai_failure("AI 评分结果格式异常，请稍后重试")
    if not isinstance(analysis, str) or not analysis.strip():
        raise _ai_failure("AI 评分结果格式异常，请稍后重试")
    return {
        "name": dimension_name,
        "score": score,
        "analysis": analysis.strip(),
        "suggestion": str(data.get("suggestion", "")).strip(),
    }


def _build_summary(dimensions: list[dict[str, Any]]) -> str:
    average_score = round(sum(int(item["score"]) for item in dimensions) / len(dimensions))
    weak_dimensions = [DIMENSION_LABELS[item["name"]] for item in dimensions if int(item["score"]) < 70]
    if weak_dimensions:
        return f"综合评分 {average_score}，建议优先补强{'、'.join(weak_dimensions)}。"
    return f"综合评分 {average_score}，简历与目标岗位整体匹配度较高。"


DIMENSION_LABELS = {
    "technology_stack": "技术栈匹配度",
    "experience": "项目/工作经验匹配度",
    "education": "学历与背景匹配度",
    "potential": "成长潜力与表达能力",
    "resume_structure": "简历结构完整性与可读性",
}

DIMENSION_PROMPTS = {
    "technology_stack": (
        "你是专业的简历评估专家和职业发展顾问。请通过深度分析岗位技能需求，对候选人技术栈进行结构化评价。"
        "必须覆盖：总体符合度、核心亮点、主要差距、核心技术栈诊断、优势技术栈、待提升技术栈、行动计划和总结。"
        "技术栈诊断必须比较岗位具体要求、简历体现、匹配度、评价与分析、提升建议。"
        "评分时重点参考关键词语义匹配上下文，包括简历关键词、岗位关键词、匹配对、整体语义相似度。"
    ),
    "experience": (
        "你是一位顶尖的职业发展顾问和简历分析专家。请深度分析项目经历、工作经历、奖项与目标岗位的匹配度。"
        "必须覆盖详细匹配度分析、基于外部参考的发展计划、项目实践建议、竞赛或行业活动建议、学习资源与社区建议。"
        "评分要体现经历与岗位职责、技术深度、业务价值、可迁移经验和结果量化程度的匹配情况。"
    ),
    "education": (
        "你是学历与背景评估专家。请根据候选人的教育经历、学校背景、专业方向、课程/竞赛/科研信息与岗位要求进行评分。"
        "如果岗位没有明确学历要求，应从学科实力、专业相关性、基础能力证明和背景可信度进行评估。"
        "分析需要说明满足点、风险点和可补充证明材料。"
    ),
    "potential": (
        "你是一名资深人才分析师和潜力评估专家。请基于简历和岗位描述，评估候选人在该岗位上的胜任潜力。"
        "分析必须超越关键词匹配，挖掘可迁移能力、学习能力、适应性、解决问题与创新思维、项目管理执行力、"
        "沟通协作、人际影响力、自我驱动和非传统但有价值的经历。"
    ),
    "resume_structure": (
        "你是专业 HR 招聘专家。请针对简历中的项目经历和工作经历进行结构化程度评价，核心依据是 STAR 法则。"
        "必须分析 Situation 情境、Task 任务、Action 行动、Result 结果四个要素，关注清晰性、具体性、个人贡献、"
        "技术决策、创新性、量化结果、影响力和可信度。"
    ),
}


def _dimension_messages(
    *,
    dimension_name: str,
    parsed_resume: ParsedResume,
    target_job: dict[str, Any],
    search_context: str,
    keyword_context: str,
) -> list[dict[str, str]]:
    system_prompt = (
        f"{DIMENSION_PROMPTS[dimension_name]}"
        "请只返回 JSON 对象，不要输出 Markdown 包裹或解释。"
        f"JSON 字段固定为 name、score、analysis、suggestion，其中 name 必须为 {dimension_name}，"
        "score 是 0-100 整数，analysis 是用户可见评分依据，suggestion 是用户可执行优化建议。"
    )
    user_prompt = (
        f"目标岗位：{target_job['title']}\n"
        f"岗位描述：{target_job['description']}\n"
        f"评分维度：{dimension_name} / {DIMENSION_LABELS[dimension_name]}\n"
        f"简历摘要：{_resume_prompt_text(parsed_resume)}\n"
        f"搜索增强参考：{search_context}\n"
        f"关键词语义匹配参考：{keyword_context}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _search_context(ctx: AbilityContext, parsed_resume: ParsedResume, target_job: dict[str, Any]) -> str:
    queries = _search_queries(parsed_resume, target_job)
    if not queries or ctx.ai_gateway is None:
        return "未获取到外部参考信息"
    snippets: list[str] = []
    for query in queries:
        result = ctx.ai_gateway.search(AiSearchRequest(query=query))
        if result.success and result.data:
            snippets.append(f"{query}：{_limit_text(str(result.data), 600)}")
    return "\n".join(snippets) if snippets else "未获取到外部参考信息"


def _keyword_semantic_context(ctx: AbilityContext, parsed_resume: ParsedResume, target_job: dict[str, Any]) -> str:
    resume_text = " ".join(
        part
        for part in (
            parsed_resume.sections.get("technology_stack", ""),
            parsed_resume.sections.get("program", ""),
            parsed_resume.sections.get("work", ""),
        )
        if part
    )
    job_text = f"{target_job.get('title', '')} {target_job.get('description', '')}"
    resume_keywords = _extract_keywords(resume_text)
    job_keywords = _extract_keywords(job_text)
    similarity = _embedding_similarity(ctx, " ".join(resume_keywords), " ".join(job_keywords))
    matched_pairs = _match_keywords(resume_keywords, job_keywords)
    return (
        f"简历关键词：{', '.join(resume_keywords) or '未提取到'}\n"
        f"岗位关键词：{', '.join(job_keywords) or '未提取到'}\n"
        f"关键词匹配对：{'; '.join(matched_pairs) or '未形成明确匹配'}\n"
        f"整体语义相似度：{similarity}"
    )


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}|[\u4e00-\u9fff]{2,}", text)
    stopwords = {
        "负责",
        "相关",
        "进行",
        "岗位",
        "目标",
        "要求",
        "项目",
        "经验",
        "熟悉",
        "掌握",
        "使用",
        "具备",
        "以及",
    }
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        normalized = token.strip()
        lowered = normalized.lower()
        if lowered in seen or normalized in stopwords:
            continue
        seen.add(lowered)
        keywords.append(normalized)
        if len(keywords) >= 15:
            break
    return keywords


def _match_keywords(resume_keywords: list[str], job_keywords: list[str]) -> list[str]:
    pairs: list[str] = []
    lowered_job = [(keyword, keyword.lower()) for keyword in job_keywords]
    for resume_keyword in resume_keywords:
        lowered_resume = resume_keyword.lower()
        exact = next((keyword for keyword, lowered in lowered_job if lowered == lowered_resume), None)
        if exact is not None:
            pairs.append(f"{resume_keyword}<->{exact}")
            continue
        partial = next(
            (
                keyword
                for keyword, lowered in lowered_job
                if lowered_resume in lowered or lowered in lowered_resume
            ),
            None,
        )
        if partial is not None:
            pairs.append(f"{resume_keyword}<->{partial}")
    return pairs[:10]


def _embedding_similarity(ctx: AbilityContext, resume_keywords: str, job_keywords: str) -> str:
    embedding = getattr(ctx.ai_gateway, "embedding", None)
    if embedding is None or not resume_keywords or not job_keywords:
        return "未启用"
    resume_result = embedding(AiEmbeddingRequest(input=resume_keywords))
    job_result = embedding(AiEmbeddingRequest(input=job_keywords))
    if not resume_result.success or not job_result.success:
        return "未获取到"
    resume_vector = _coerce_vector(resume_result.data)
    job_vector = _coerce_vector(job_result.data)
    if not resume_vector or not job_vector or len(resume_vector) != len(job_vector):
        return "未获取到"
    dot = sum(left * right for left, right in zip(resume_vector, job_vector, strict=True))
    left_norm = math.sqrt(sum(item * item for item in resume_vector))
    right_norm = math.sqrt(sum(item * item for item in job_vector))
    if left_norm == 0 or right_norm == 0:
        return "未获取到"
    return f"{dot / (left_norm * right_norm):.2f}"


def _coerce_vector(value: Any) -> list[float]:
    if isinstance(value, dict):
        value = value.get("embedding") or value.get("vector") or value.get("output")
    if not isinstance(value, list):
        return []
    vector: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            return []
        vector.append(float(item))
    return vector


def _search_queries(parsed_resume: ParsedResume, target_job: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    title = str(target_job.get("title", "")).strip()
    description = str(target_job.get("description", "")).strip()
    if title or description:
        queries.append(_limit_text(f"{title} {description} 核心技能要求 行业评价", 160))
    school = parsed_resume.sections.get("school", "").strip()
    if school:
        queries.append(_limit_text(f"{school} 院校背景 学科实力 排名", 160))
    experience = "\n".join(
        part
        for part in (
            parsed_resume.sections.get("program", "").strip(),
            parsed_resume.sections.get("work", "").strip(),
        )
        if part
    )
    if experience:
        queries.append(_limit_text(f"{title} {experience} 项目经验 匹配度", 160))
    return queries[:3]


def _resume_prompt_text(parsed_resume: ParsedResume) -> str:
    section_labels = {
        "technology_stack": "技能栈",
        "program": "项目经历",
        "work": "工作经历",
        "school": "教育经历",
        "awards": "荣誉奖项",
        "school_work": "校园/社团经历",
    }
    parts: list[str] = []
    for key, label in section_labels.items():
        value = parsed_resume.sections.get(key, "").strip()
        if value:
            parts.append(f"{label}：{_limit_text(value, 1200)}")
    if not parts:
        parts.append(f"简历全文：{_limit_text(parsed_resume.full_text, 3000)}")
    return "\n".join(parts)


def _limit_text(value: str, max_length: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length] + "..."
