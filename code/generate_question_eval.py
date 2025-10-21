import asyncio
import os
import shutil

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

import generate_question_answer
import multimoding_dispose
from base import llm_qwen
from do_record_video import MediaRecorder
from multimoding_dispose import body_movement_analyse_full_flow, emotion_analyse_full_flow
from state import InterviewState


async def get_answer_quality_eval(question:str,job:str,resume:str,answer:str)->dict:
    print("正在进行问题答案评估")
    system_prompt_template = """
# Role: 面试评估专家 (Interview Assessment Expert)

## 🎯 **你的任务 (Your Mission)**

你是一名顶级的面试评估专家，拥有多年技术和管理招聘经验。你的核心任务是基于我提供的候选人信息、职位要求、面试问题和候选人的回答，进行一次专业、深入、公正的评估。你需要严格遵循我为你定义的评估框架和标准，输出对用户答案进行评分。

---

## 📝 **评估框架与标准 (Evaluation Framework & Standards)**

你必须严格遵循以下标准。在评估前，首先判断问题属于哪一类，然后应用该类的具体标准。

### **第一步：判断问题类型 (Step 1: Classify the Question Type)**
从以下四种类型中选择一个最匹配的：
1.  **技术基础知识考察 (Technical Fundamentals)**
2.  **项目经历考察 (Project Experience)**
3.  **业务经历/场景题考察 (Business Sense / Case Study)**
4.  **软技能考察 (Soft Skills)**

### **第二步：应用评估标准进行分析 (Step 2: Apply Evaluation Criteria for Analysis)**

#### 1. 技术基础知识考察 (Technical Fundamentals)
*   **核心评估点：** 候选人是否具备扎实的、与职位要求匹配的核心技术知识、原理理解和基本技能。
*   **评估标准：**
    *   **准确性 (Accuracy):** 概念、术语、原理、语法是否正确？
    *   **深度与广度 (Depth & Breadth):** 对核心概念的理解是否深入（如原理、机制、优缺点）？知识面是否覆盖关键领域？
    *   **理解与应用 (Understanding & Application):** 是死记硬背还是真正理解？能否用自己的话解释并应用于简单场景？
    *   **清晰度与表达 (Clarity & Expression):** 解释是否简洁、有逻辑、专业？
    *   **最新动态意识 (Up-to-date Awareness - Bonus):** 是否了解相关技术的最新趋势？

#### 2. 项目经历考察 (Project Experience)
*   **核心评估点：** 候选人过去实际工作的真实性、深度、贡献度以及从项目中学习和成长的能力。
*   **评估标准 (STAR原则)：**
    *   **情景 (Situation):** 项目背景、目标、挑战是否清晰？
    *   **任务 (Task):** 个人角色和职责是否明确？
    *   **行动 (Action):**
        *   **技术深度：** 具体做了什么？技术选型原因？如何解决关键问题？
        *   **工程实践：** 是否体现了良好的代码规范、测试、版本控制等实践？
        *   **协作沟通：** 如何与团队协作？
    *   **结果 (Result):**
        *   **成果量化：** 个人贡献和项目成果是否用具体指标量化（如性能提升X%，成本下降Y%）？
        *   **业务影响：** 工作对业务的实际影响是什么？
    *   **反思与学习 (Reflection & Learning):** 有何成功经验、失败教训？如果重来如何改进？学到了什么？

#### 3. 业务经历/场景题考察 (Business Sense / Case Study)
*   **核心评估点：** 理解业务需求、将技术与业务结合、解决复杂开放性问题的逻辑思维能力。
*   **评估标准：**
    *   **业务理解 (Business Acumen):** 是否准确理解问题背后的业务背景、目标和核心需求？
    *   **问题定义 (Problem Framing):** 能否清晰界定问题的核心与边界？
    *   **分析框架 (Analytical Framework):** 思维是否结构化？能否将问题拆解并判断优先级？
    *   **解决方案 (Solution):**
        *   **关联性：** 技术方案是否紧密服务于业务目标？
        *   **可行性：** 是否考虑了技术、资源、时间的限制和风险？
        *   **数据驱动 (Data-Driven):** 是否考虑用数据指标来衡量效果？
    *   **沟通与应变 (Communication & Adaptability):** 能否清晰阐述方案？被挑战时能否灵活调整？

#### 4. 软技能考察 (Soft Skills)
*   **核心评估点：** 沟通协作、学习能力、问题解决方式、职业素养和文化匹配度。
*   **评估标准 (贯穿回答的始终)：**
    *   **沟通能力 (Communication):** 表达是否清晰、有条理？是否能积极倾听？
    *   **协作能力 (Collaboration):** 是否体现团队意识和处理分歧的能力？
    *   **学习能力与成长心态 (Learnability & Growth Mindset):** 是否表现出好奇心、反思总结和适应性？
    *   **解决问题能力 (Problem-Solving):** 面对模糊问题时，分析是否具备逻辑性和系统性？
    *   **主动性/责任感 (Proactiveness/Ownership):** 是否表现出主动承担责任、推动进展的特质？
    *   **职业素养 (Professionalism):** 回答是否诚实？态度是否积极专业？

---
## 📤 **输出格式 (Output Format)**

请严格按照json格式生成答案， 存在两个键值 'score','eval' 
'score' 对应为 int 类型，表示你对答案生成的评分 （满分100分）
'eval' 对应类型为 string 类型  要包含以下方面：

*1. 优点分析 (Strengths)**
*   `[基于评估标准，分点列出候选人回答中的优点。例如：技术概念阐述准确，对XXX原理理解深入。]`
*   `[项目描述中，能够清晰运用STAR原则，量化指标明确。]`
*   `[...更多优点]`

**2. 待改进点分析 (Areas for Improvement)**
*   `[基于评估标准，分点列出候选人回答中的不足之处。例如：对技术选型背后的“为什么”解释不足，未能体现出权衡过程。]`
*   `[在描述项目贡献时，个人职责与团队成果有所混淆，个人贡献不够突出。]`
*   `[...更多待改进点]`

```
            """
    system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            ('user', """
                        询问问题 ：
                        {question}
                        应聘岗位
                        {job}
                        用户简历：
                        {resume}
                        候选人回答
                        {answer}

                """)
        ]
    )
    chain = prompt | llm_qwen | JsonOutputParser()
    ans = await chain.ainvoke({'question':question,'job':job,'resume':resume,'answer':answer})
    print(ans['score'])
    print(ans['eval'])
    if isinstance(ans, dict):
        print("答案评估结束")
        return ans
    else:
        return {'score':60}

async def do_answer(resume: InterviewState) -> dict:

    print('正在获取问题')
    question = resume['interview_question']
    path_list = ['面试视频（用户）', '语音资料（用户）']
    print(f'获取到的问题为：{question}')
    for folder_path in path_list:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    for i, q in enumerate(question, 1):
        video_base_path = '面试视频（用户）'
        record_base_path = '语音资料（用户）'
        print(f'{i}.{q}\n\n系统将自动进行录音录像，请做好准备。')
        video_path = os.path.join(video_base_path, f'{i}.mp4')
        record_path = os.path.join(record_base_path, f'{i}.mp3')
        recorder = MediaRecorder(record_path, video_path)
        recorder.start_recording()
        input('如果问题回答完毕，请点击控制台然后按回车键就会停止录制')
        recorder.stop_recording()

async def get_user_question_answer(resume: InterviewState) -> dict:
    print('正在获取问题答案')
    task = []
    input_path = '语音资料（用户）'
    paths = os.listdir(input_path)
    path = sorted(
        paths,
        key=lambda x: os.path.getctime(os.path.join(input_path, x)),
    )
    path = [os.path.join(input_path, p) for p in path]
    for p in path:
        task.append(asyncio.create_task(multimoding_dispose.audio_text(p)))
    result = await asyncio.gather(*task)
    print(result)
    return {'answer': result}

async def get_answer_eval(resume: InterviewState) -> dict:
    print('正在进行答案评估')
    num = sum(resume['interview_question_num'])
    print(num)
    base_technology_num = resume['interview_question_num'][0]
    job = resume['job']
    answer = resume['answer']
    program_num = resume['interview_question_num'][1] + resume['interview_question_num'][2]
    soft_question_num = resume['interview_question_num'][3]
    question = resume['interview_question']
    resume = resume['resume']['work'] + resume['resume']['program'] + resume['resume']['technology_stack']
    task = []
    for i in range(base_technology_num):
        task.append(generate_question_answer.get_base_technology_answer(question[i], job))
    for i in range(program_num):
        task.append(
            generate_question_answer.get_program_question_answer(question[base_technology_num + i], job, resume))
    for i in range(soft_question_num):
        task.append(
            generate_question_answer.get_soft_question_answer(question[base_technology_num + program_num + i], job,
                                                              resume))
    for i, ans in enumerate(answer):
        task.append(get_answer_quality_eval(question[i], job, resume, answer[i]))
    task.append(body_movement_analyse_full_flow())
    task.append(emotion_analyse_full_flow())
    result = await asyncio.gather(*task)
    result1 = result[:num]
    result2 = result[num:len(task) - 2]
    result_3 = result[-2]  # 肢体动作
    result_4 = result[-1]  # 语音语调
    score = 0
    # 初始化结果结构
    r = {
        'analyse': [],
        'standard_answer': [],
        'eval': [],
        'answer': answer,
        'answer_score': []
    }

    # 遍历结果并填充数据
    for i, h in zip(result1, result2):
        r['analyse'].append(i['analysis'])
        r['standard_answer'].append(i['answer'])
        r['eval'].append(h['eval'])
        score += h['score']

    # 计算综合评分
    print(result_3)
    print(result_4)
    r['answer_score'] = [
        score / len(result1) if result1 else 0,
        sum(result_3) / len(result_3) if result_3 else 0,
        sum(result_4) / len(result_4) if result_4 else 0
    ]
    print(r)
    return r

