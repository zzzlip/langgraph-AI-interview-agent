from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from openai import OpenAI,AsyncOpenAI
import api_key
class Resume(TypedDict):
    messages: List
    page: Annotated[str, '判断阶段用于指引工作流']
    code_id: Annotated[str, '用户在codeforce的账号名字']
    picture_path: Annotated[str, '用户提供的用于生成简历照片的存放路径']
    resume_radar_path: Annotated[str, '生成的雷达图的保存路径']
    resume_text: Annotated[str, '简历的全部内容']
    path: Annotated[str, '用户简历存储路径']
    job: Annotated[str, '用户面试岗位的具体信息']
    program: Annotated[str, '简历中体现的项目经历']
    work: Annotated[str, '简历中体现的实习/工作/学校经历']
    school: Annotated[str, '简历中体现的学校背景']
    school_work: Annotated[str,'在学校参加的活动，包括志愿活动，学生会等等']
    technology_stack: Annotated[str, '简历中体现的技术栈/个人优势']
    awards: Annotated[str, '简历中体现的奖项证书']
    technology_stack_evaluate: Annotated[str, '对其技术栈/个人优势与岗位的匹配评价与建议']
    technology_stack_score: Annotated[float, '对其技术栈/个人优势与岗位的匹配分数']
    resume_struct_evaluate: Annotated[str, '对简历书写结构的评价与建议']
    resume_struct_score: Annotated[float, '对简历书写结构的评评分']
    experience_evaluate: Annotated[str, '工作经验与岗位适配性的评价']
    experience_score: Annotated[float, '工作/实习经验与岗位适配性的评分']
    school_score: Annotated[float, '学历背景与岗位适配性的评分']
    potential_evaluate: Annotated[str, '简历中展示的潜力评价']
    potential_score: Annotated[float, '简历中展示的潜力评分']
    optimize_resume_path: Annotated[str, '优化的简历存放路径']
    report_path: Annotated[str, '简历评估报告存放路径']
    interview_question_num: Annotated[list,'存储面试基础知识问题/项目问题/业务问题/软技能问题相对应所出题目数量']
    question: Annotated[list, '存储所出相关问题']
    standard_answer:Annotated[list,'存储相关问题的标准答案']
    analyse:Annotated[list,'存储所出问题对应的分析结果']
    eval:Annotated[list,'存储对用户答案的分析']
    answer:Annotated[list,'存储用户对问题的答案']
    answer_score:Annotated[list,'存储用户答案质量/肢体动作/语音语调情绪得分']


llm_google = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key.google_api,
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True,
)
llm_reasoner = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key.dp_api,
    model="deepseek-reasoner",
    temperature=0.7,
    streaming=True,
)
llm=ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key.dp_api,
    model="deepseek-chat",
    temperature=0.7,
    streaming=True,
)
llm_qwen=ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key.qwen_api,
    model="qwen-max-latest",
    temperature=0.7,
    streaming=True,
)
video_client= AsyncOpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key.qwen_api,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
# llm_google=llm_qwen
# llm_reasoner=llm_qwen
# llm=llm_qwen