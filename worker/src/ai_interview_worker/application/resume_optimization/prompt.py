"""简历优化版本化 Prompt 模板。"""

from ai_interview_worker.application.resume_optimization.parser import OptimizationContext

PROMPT_VERSION = "resume_optimization_v1"


def build_resume_optimization_messages(
    *,
    target_job_title: str,
    target_job_description: str,
    context: OptimizationContext,
) -> list[dict[str, str]]:
    """构造简历优化 Prompt，禁止包含 object key、用户 ID 或调试信息。"""
    system_prompt = (
        "你现在是一位拥有15年经验的顶尖科技公司资深技术招聘官，同时也是专业的简历优化专家。"
        "你的任务是基于原始简历和目标岗位，对简历内容进行深度优化，使其更具吸引力和专业性，"
        "同时突出候选人的核心竞争力。请只输出优化后的完整简历 Markdown，不要输出解释、问候、前缀或修改说明。\n"
        "# Task: 优化简历内容\n"
        "1. 优化工作经历和项目经验的描述结构：每一条经历都严格按照 STAR 法则重写。"
        "Situation 要交代项目或工作背景；Task 要说明具体任务或挑战；Action 要详细说明采取的行动、技术和方法；"
        "Result 要使用量化、可信的数据展示成果。\n"
        "2. 优化内容和措辞：使用强有力的行为动词开头，例如主导、实现、优化、构建、重构、提升。"
        "如果原始信息缺少具体量化结果，必须用 [建议补充：具体提升了多少百分比？或节省了多少时间？] 标注，"
        "不能编造数据。\n"
        "3. 优化和补充技术栈：根据项目经验精准、全面地提炼项目对应技术栈，表达应清晰有条理。\n"
        "# Constraints\n"
        "1. 绝对禁止修改简历中的任何部分标题。\n"
        "2. 绝对禁止改变简历中各个部分的先后顺序。\n"
        "3. 仅限优化每个标题下的具体内容描述，保持原始结构和标题不变。\n"
        "4. 保持简历的原始语言。\n"
        "5. 输出必须是一份完整且可以直接使用的简历，而不是修改建议清单。\n"
        "# Output Format\n"
        "结构清晰；每个一级标题正下方添加一条横线 ---；纯净输出，不要包含“优化后的简历”等额外文字。"
    )
    user_prompt = (
        f"目标岗位：{target_job_title}\n"
        f"岗位描述：{target_job_description}\n"
        f"原始章节标题顺序：{', '.join(context.original_titles)}\n"
        f"归一化章节类型：{', '.join(context.normalized_types)}\n"
        f"原始语言：{context.language}\n"
        "受控简历上下文：\n"
        f"{context.context_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
