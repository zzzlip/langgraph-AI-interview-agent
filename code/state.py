from operator import add
from typing import Annotated, List, TypedDict
from langgraph.graph import MessagesState
from pydantic import BaseModel


class ResumeContent(TypedDict):
    resume_text: Annotated[str, '简历的全部内容']
    program: Annotated[str, '简历中体现的项目经历']
    work: Annotated[str, '简历中体现的实习/工作/学校经历']
    school: Annotated[str, '简历中体现的学校背景']
    school_work: Annotated[str, '在学校参加的活动，包括志愿活动，学生会等等']
    technology_stack: Annotated[str, '简历中体现的技术栈/个人优势']
    awards: Annotated[str, '简历中体现的奖项证书']
    job: Annotated[str, '用户面试岗位的具体信息']

class ResumeEvaluateInPut(TypedDict):
    resume: Annotated[ResumeContent, '存放的是解析后的简历内容']

class ResumeEvaluateOutPut(TypedDict):
    page: Annotated[bool, '判断简历是否通过初筛']

class ResumeEvaluateState(BaseModel):
    technology_stack_evaluate: Annotated[str, '对其技术栈/个人优势与岗位的匹配评价与建议']
    technology_stack_score: Annotated[float, '对其技术栈/个人优势与岗位的匹配分数']
    resume_struct_evaluate: Annotated[str, '对简历书写结构的评价与建议']
    resume_struct_score: Annotated[float, '对简历书写结构的评评分']
    experience_evaluate: Annotated[str, '工作经验与岗位适配性的评价']
    experience_score: Annotated[float, '工作/实习经验与岗位适配性的评分']
    school_score: Annotated[float, '学历背景与岗位适配性的评分']
    potential_evaluate: Annotated[str, '简历中展示的潜力评价']
    potential_score: Annotated[float, '简历中展示的潜力评分']
    resume_radar_path: Annotated[str, '生成的雷达图的保存路径']
    report_path: Annotated[str, '简历评估报告存放路径']
    page: Annotated[bool, '判断简历是否通过初筛']



class CodeQuestionOutput(BaseModel):
    tag: Annotated[List[str], '表示所要抽取的题目对应的标签，每一类只给出一个标签。']
    score: Annotated[list[int], '表示对应标签所出题目的难度分数,每一个标签只用一个数字表示即可']
    count: Annotated[list[int], '表示对应标签所出题目个数，一定要注意总体题目数量要等于5']


class CodeTestInput(TypedDict):
    position: Annotated[str, '应聘职位信息']
    code_id: Annotated[str, '用户在codeforce的账号名字']

class CodeTestOutPut(TypedDict):
    code_page: Annotated[bool, '判断是否通过笔试']


class CodeState(TypedDict):
    question_class: Annotated[CodeQuestionOutput, '选取题目的相关难度系数']
    question: Annotated[list[dict], "选取的题目", add]
    code_id: Annotated[str, '用户在code_force的账号名字']
    position: str  # 将 position 也加入状态，以便后续节点可能使用


class InterviewState(TypedDict):
    interview_question_num: Annotated[list, '存储面试基础知识问题/项目问题/业务问题/软技能问题相对应所出题目数量']
    interview_question: Annotated[list, '存储所出相关问题',add]
    standard_answer: Annotated[list, '存储相关问题的标准答案']
    analyse: Annotated[list, '存储所出问题对应的分析结果']
    eval: Annotated[list, '存储对用户答案的分析']
    answer: Annotated[list, '存储用户对问题的答案']
    answer_score: Annotated[list, '存储用户答案质量/肢体动作/语音语调情绪得分']
    job: Annotated[str, '用户面试岗位的具体信息']
    resume: Annotated[ResumeContent, '存放的是解析后的简历内容']
class InterviewInPut(TypedDict):
    job: Annotated[str, '用户面试岗位的具体信息']
    resume: Annotated[ResumeContent, '存放的是解析后的简历内容']
class InterviewOutPut(TypedDict):
    answer_score: Annotated[list, '存储用户答案质量/肢体动作/语音语调情绪得分']
class MainState(MessagesState):
    workflow_step:Annotated[str,'用于指引工作流']
    page: Annotated[bool, '用于判断是否通过某项测试']
    picture_path: Annotated[str, '用户提供的用于生成简历照片的存放路径']
    path: Annotated[str, '用户简历存储路径']
    job: Annotated[str, '用户面试岗位的具体信息']
    code_id: Annotated[str, '用户在code_force的账号名字']
    resume:Annotated[ResumeContent,'存放的是解析后的简历内容']
    optimize_resume_path: Annotated[str, '优化的简历存放路径']
    answer_score: Annotated[list, '存储用户答案质量/肢体动作/语音语调情绪得分']

