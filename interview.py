import os
import shutil

import generate_interview_question
import multimoding_dispose
import generate_question_answer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import asyncio
from multimoding_dispose import body_movement_analyse_full_flow,emotion_analyse_full_flow
from base import Resume,llm
from do_record_video import MediaRecorder


def get_question_num(resume:Resume)->dict:
    job=resume['job']
    prompt_template="""
    # 角色
你是一位经验丰富的HR技术顾问（HR Tech Consultant），同时也是一名资深的面试官。你擅长分析职位描述（JD），精准判断岗位所需的核心能力，并以此为依据设计科学的面试题目结构。

# 任务
你的核心任务是：分析我提供的【应聘岗位信息】，判断该岗位是“技术驱动型”还是“业务驱动型”，并根据你的判断，为总数为5道题的一面环节，分配合理的题目数量。

# 背景信息
我正在开发一个自动化一面系统。系统需要根据不同的岗位信息，智能地生成面试题目。一面总共包含5道题，题型分为三类：
1.  **技术基础知识问答**：考察候选人对基础理论、工具、算法，技术栈等的掌握程度
2.  **项目经历技术问答**：深入挖掘候选人过去项目中的技术实践、架构设计和问题解决能力。
3.  **业务问答**：考察候选人对相关业务领域的理解、产品感知和商业思维。

# 工作流程
1.  **分析岗位信息**：仔细阅读我提供的【应聘岗位信息】，理解其职责、要求和关键词。
2.  **评估岗位重心**：
    *   如果JD中频繁出现底层原理、架构设计、算法、性能优化、源码等关键词，则判断为“**技术驱动型**”。
    *   如果JD中更侧重于行业解决方案、客户需求、产品功能、数据分析、业务流程等关键词，则判断为“**业务驱动型**”。
    *   如果两者兼备，请根据主要职责判断其更偏向哪一方。
3.  **分配题目数量**：根据你的判断，为上述三类题型分配题目数量。

# 约束与规则
1.  题目总数 **必须** 为 6 道。
2. “技术基础知识” “业务问答” **至少** 分配 1 道题， “项目经历技术问答”至少分配两道、。
3.  最终的分配方案需要体现出你对岗位重心的判断。例如，“技术驱动型”岗位的技术基础知识类题目数目应多于业务问题，反之则基础知识类题目数目应少于业务问题岗位。

# 输出格式
请严格按照以下json格式输出，里面存在一个键 one_interview_question_num 
interview_question_num 对应类型为 list 对应存储着基础知识问题/项目问题/业务问题相对应所出题目数量
岗位信息：
{job}
    """
    prompt=ChatPromptTemplate.from_template(prompt_template)
    chain=prompt|llm|JsonOutputParser()
    result=chain.invoke({'job':job})
    print(result)
    if isinstance(result,dict):
        result['interview_question_num']=result['interview_question_num']+[0]
        result['page']='一面'
        return  result
    return {'interview_question_num':[2,2,2,0],'page':'一面'}


async def get_interview_question(resume:Resume)->dict:
    questions = await asyncio.gather(
        generate_interview_question.generate_technology_question(resume),
        generate_interview_question.generate_program_question(resume),
        generate_interview_question.generate_business_question(resume),
        generate_interview_question.generate_soft_power_question(resume)
    )
    question = []
    print(questions)
    try:
        for i in questions:
            question += i['question']
    except Exception as e:
        print(e)
    return {'question':question}
async def do_answer(resume:Resume)->dict:

    print('正在获取问题')
    question=resume['question']
    path_list=['面试视频（用户）', '语音资料（用户）']
    print(f'获取到的问题为：{question}')
    for folder_path in path_list:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    for i,q in enumerate(question,1):
        video_base_path = '面试视频（用户）'
        record_base_path = '语音资料（用户）'
        print(f'{i}.{q}\n\n系统将自动进行录音录像，请做好准备。')
        video_path=os.path.join(video_base_path, f'{i}.mp4')
        record_path=os.path.join(record_base_path, f'{i}.mp3')
        recorder = MediaRecorder(record_path, video_path)
        recorder.start_recording()
        input('如果问题回答完毕，请点击控制台然后按回车键就会停止录制')
        recorder.stop_recording()



async def get_question_answer(resume:Resume)->dict:
    print('正在获取问题答案')
    task=[]
    input_path='语音资料（用户）'
    paths=os.listdir(input_path)
    path = sorted(
        paths,
        key=lambda x: os.path.getctime(os.path.join(input_path, x)),
    )
    path=[os.path.join(input_path, p) for p in path]
    for p in path:
        task.append(asyncio.create_task(multimoding_dispose.audio_text(p)))
    result=await asyncio.gather(*task)
    print(result)
    return {'answer': result}

async def get_answer_eval(resume:Resume)->dict:
    print('正在进行答案评估')
    num=sum(resume['interview_question_num'])
    print(num)
    base_technology_num=resume['interview_question_num'][0]
    job=resume['job']
    answer=resume['answer']
    program_num=resume['interview_question_num'][1]+resume['interview_question_num'][2]
    soft_question_num=resume['interview_question_num'][3]
    question=resume['question']
    resume=resume['work']+resume['program']+resume['technology_stack']
    task=[]
    for i in range(base_technology_num):
        task.append(generate_question_answer.get_base_technology_answer(question[i],job))
    for i in range(program_num):
        task.append(generate_question_answer.get_program_question_answer(question[base_technology_num+i],job,resume))
    for i in range(soft_question_num):
        task.append(generate_question_answer.get_soft_question_answer(question[base_technology_num+program_num+i],job,resume))
    for i,ans in enumerate(answer):
        task.append(generate_question_answer.get_answer_quality_eval(question[i],job,resume,answer[i]))
    task.append(body_movement_analyse_full_flow())
    task.append(emotion_analyse_full_flow())
    result=await asyncio.gather(*task)
    result1=result[:num]
    result2=result[num:len(task)-2]
    result_3=result[-2] #肢体动作
    result_4=result[-1] # 语音语调
    score=0
    # 初始化结果结构
    r = {
        'analyse': [],
        'standard_answer': [],
        'eval': [],
        'answer': answer,
        'answer_score':[]
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












