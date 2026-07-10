"""简历优化结构化摘要构造。"""

from ai_interview_worker.application.resume_optimization.parser import ResumeDocument

ALLOWED_OPTIMIZED_SECTIONS = {
    "personal_summary",
    "education",
    "work_experience",
    "project_experience",
    "technology_stack",
    "awards",
    "other",
}

SECTION_DIRECTIONS = {
    "personal_summary": "强化个人总结与目标岗位匹配度",
    "education": "保留教育背景结构并优化表达清晰度",
    "work_experience": "按 STAR 法则重写工作经历并突出量化成果",
    "project_experience": "按 STAR 法则优化项目经验并补充业务结果",
    "technology_stack": "结合项目经历补强技术栈表达",
    "awards": "梳理奖项信息并突出可信背书",
    "other": "保留补充信息并优化措辞",
}


def build_result_summary(
    *,
    document: ResumeDocument,
    avatar_used: bool,
    prompt_version: str,
    template: str,
) -> dict[str, object]:
    """构造脱敏、结构化的简历优化摘要。"""
    optimized_sections = _optimized_sections(document)
    optimization_items = [
        {
            "section": section,
            "direction": SECTION_DIRECTIONS[section],
        }
        for section in optimized_sections
    ]
    return {
        "schema_version": 1,
        "result_type": "resume_optimization",
        "summary": _summary_text(optimized_sections),
        "optimized_sections": optimized_sections,
        "optimization_items": optimization_items,
        "avatar_used": avatar_used,
        "avatar_optional": True,
        "prompt_version": prompt_version,
        "template": template,
    }


def _optimized_sections(document: ResumeDocument) -> list[str]:
    seen = set()
    sections: list[str] = []
    for section in document.sections:
        normalized_type = section.normalized_type
        if normalized_type in ALLOWED_OPTIMIZED_SECTIONS and normalized_type not in seen:
            seen.add(normalized_type)
            sections.append(normalized_type)
    return sections


def _summary_text(optimized_sections: list[str]) -> str:
    if not optimized_sections:
        return "已完成简历内容优化。"
    labels = {
        "personal_summary": "个人总结",
        "education": "教育背景",
        "work_experience": "工作经历",
        "project_experience": "项目经历",
        "technology_stack": "技术栈",
        "awards": "奖项",
        "other": "补充信息",
    }
    visible_sections = "、".join(labels[section] for section in optimized_sections)
    return f"已优化{visible_sections}，重点强化岗位匹配、STAR 表达、技术栈和成果呈现。"
