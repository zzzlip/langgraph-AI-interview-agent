"""简历优化解析、章节识别与上下文裁剪。"""

from dataclasses import dataclass

from ai_interview_worker.legacy_adapter.resume_optimization import extract_resume_text

SECTION_TYPE_BY_TITLE = {
    "个人总结": "personal_summary",
    "个人优势": "personal_summary",
    "自我评价": "personal_summary",
    "教育经历": "education",
    "教育背景": "education",
    "教育": "education",
    "工作经历": "work_experience",
    "工作经验": "work_experience",
    "实习经历": "work_experience",
    "项目经历": "project_experience",
    "项目经验": "project_experience",
    "项目": "project_experience",
    "专业技能": "technology_stack",
    "技能栈": "technology_stack",
    "技能": "technology_stack",
    "技术栈": "technology_stack",
    "荣誉奖项": "awards",
    "获奖情况": "awards",
    "奖项": "awards",
}

SECTION_PRIORITY = {
    "work_experience": 0,
    "project_experience": 1,
    "technology_stack": 2,
    "personal_summary": 3,
    "education": 4,
    "awards": 5,
    "other": 6,
}
UNKNOWN_SECTION_TITLES = {"其他", "其他经历", "附加信息", "补充信息"}


@dataclass(frozen=True)
class ResumeSection:
    """简历章节。"""

    title: str
    normalized_type: str
    content: str
    order: int


@dataclass(frozen=True)
class ResumeDocument:
    """解析后的简历文档。"""

    full_text: str
    sections: list[ResumeSection]
    language: str


@dataclass(frozen=True)
class OptimizationContext:
    """传给优化阶段的受控上下文。"""

    context_text: str
    original_titles: list[str]
    normalized_types: list[str]
    language: str


def parse_resume_document(file_bytes: bytes, file_type: str) -> ResumeDocument:
    """解析简历 bytes，保留原始章节标题和顺序。"""
    if not isinstance(file_bytes, bytes):
        raise TypeError("file_bytes must be bytes")
    text = _normalize_text(extract_resume_text(file_bytes, file_type))
    if not _has_effective_text(text) or len("".join(text.split())) < 100:
        raise ValueError("empty resume text")
    sections = _extract_sections(text)
    return ResumeDocument(full_text=text, sections=sections, language=_detect_language(text))


def build_optimization_context(document: ResumeDocument, max_chars: int) -> OptimizationContext:
    """按章节优先级裁剪上下文，保留标题和顺序约束。"""
    if max_chars < 200:
        raise ValueError("max_chars is too small")
    selected = _select_sections(document.sections, max_chars)
    selected_by_order = sorted(selected, key=lambda section: section.order)
    context_parts = [_format_section(section) for section in selected_by_order]
    context_text = "\n\n".join(context_parts)
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars].rstrip()
    return OptimizationContext(
        context_text=context_text,
        original_titles=[section.title for section in document.sections],
        normalized_types=[section.normalized_type for section in document.sections],
        language=document.language,
    )


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _has_effective_text(text: str) -> bool:
    return bool(text) and any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in text)


def _extract_sections(text: str) -> list[ResumeSection]:
    lines = text.split("\n")
    header_indexes: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        title = line.strip().strip("：:")
        if not title:
            continue
        normalized_type = SECTION_TYPE_BY_TITLE.get(title)
        if normalized_type is not None:
            header_indexes.append((index, title, normalized_type))
        elif _looks_like_unknown_header(line):
            header_indexes.append((index, title, "other"))
    if not header_indexes:
        return [ResumeSection(title="原始简历", normalized_type="other", content=text, order=0)]

    sections: list[ResumeSection] = []
    for order, (line_index, title, normalized_type) in enumerate(header_indexes):
        next_index = header_indexes[order + 1][0] if order < len(header_indexes) - 1 else len(lines)
        content = "\n".join(lines[line_index + 1:next_index]).strip()
        sections.append(ResumeSection(title=title, normalized_type=normalized_type, content=content, order=order))
    return sections


def _looks_like_unknown_header(line: str) -> bool:
    stripped = line.strip().strip("：:")
    return stripped in UNKNOWN_SECTION_TITLES


def _select_sections(sections: list[ResumeSection], max_chars: int) -> list[ResumeSection]:
    selected: list[ResumeSection] = []
    used = 0
    priority_section_types = {"work_experience", "project_experience", "technology_stack"}
    priority_sections = [section for section in sections if section.normalized_type in priority_section_types]
    if priority_sections:
        per_section_budget = max(80, max_chars // len(priority_sections))
        for section in priority_sections:
            section_text = _format_section(section)
            if len(section_text) > per_section_budget:
                truncated_content = section.content[: max(0, per_section_budget - len(section.title) - 2)].rstrip()
                section = ResumeSection(
                    title=section.title,
                    normalized_type=section.normalized_type,
                    content=truncated_content,
                    order=section.order,
                )
                section_text = _format_section(section)
            selected.append(section)
            used += len(section_text)

    for section in sorted(sections, key=lambda item: (SECTION_PRIORITY.get(item.normalized_type, 99), item.order)):
        if section in selected or section.normalized_type in priority_section_types:
            continue
        section_text = _format_section(section)
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(section_text) <= remaining:
            selected.append(section)
            used += len(section_text)
    if not selected and sections:
        first = sections[0]
        selected.append(
            ResumeSection(
                title=first.title,
                normalized_type=first.normalized_type,
                content=first.content[: max_chars - len(first.title) - 2].rstrip(),
                order=first.order,
            )
        )
    return selected


def _format_section(section: ResumeSection) -> str:
    return f"{section.title}\n{section.content}".strip()


def _detect_language(text: str) -> str:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_chars = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    if chinese_chars and latin_chars and min(chinese_chars, latin_chars) / max(chinese_chars, latin_chars) >= 0.6:
        return "mixed"
    return "zh-CN" if chinese_chars >= latin_chars else "en-US"
