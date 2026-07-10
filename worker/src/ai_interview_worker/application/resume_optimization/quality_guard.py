"""简历优化结果结构质量校验。"""

from ai_interview_worker.application.resume_optimization.parser import OptimizationContext

EXPLANATION_PREFIXES = (
    "以下是",
    "优化后的简历",
    "当然",
    "好的",
    "Here is",
    "Below is",
)


def validate_optimized_markdown(markdown: str, context: OptimizationContext, *, min_length_ratio: float) -> None:
    """校验优化结果是否保留标题、顺序、语言和基本完整性。"""
    cleaned = markdown.strip()
    if not cleaned:
        raise ValueError("optimized markdown is empty")
    if _has_explanation_prefix(cleaned):
        raise ValueError("optimized markdown has explanation prefix")
    output_titles = _extract_output_titles(cleaned, context.original_titles)
    if output_titles != context.original_titles:
        raise ValueError("resume section titles changed")
    if _detect_language(cleaned) != context.language:
        raise ValueError("resume language changed")
    if len(cleaned) < int(len(context.context_text) * min_length_ratio):
        raise ValueError("optimized markdown is too short")


def _extract_output_titles(markdown: str, original_titles: list[str]) -> list[str]:
    original_title_set = set(original_titles)
    titles: list[str] = []
    for line in markdown.splitlines():
        normalized = line.strip().strip("#").strip().strip("：:")
        if normalized in original_title_set:
            titles.append(normalized)
    return titles


def _has_explanation_prefix(value: str) -> bool:
    head = value.lstrip()[:40]
    return any(head.startswith(prefix) for prefix in EXPLANATION_PREFIXES)


def _detect_language(text: str) -> str:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_chars = sum(1 for char in text if "a" <= char.lower() <= "z")
    if chinese_chars and latin_chars and min(chinese_chars, latin_chars) / max(chinese_chars, latin_chars) >= 0.6:
        return "mixed"
    return "zh-CN" if chinese_chars >= latin_chars else "en-US"
