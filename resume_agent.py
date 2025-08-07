import asyncio
import time

import resume_analyse
from langgraph.constants import START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langgraph.graph import StateGraph
from typing import Literal
from base import Resume
import generate_doc
from interview import get_interview_question, get_question_answer, get_question_num, get_answer_eval,do_answer
from generate_doc import create_interview_question_analyse_report
from resume_analyse import get_potential_score, get_school_score, get_resume_key_score, get_experience_match_score, get_program_struct_score
from code_test import Codetest
code=Codetest(account='')
parallel_nodes = [
        "get_potential_score",
        "get_school_score",
        "get_resume_key_score",
        "get_experience_match_score",
        "get_program_struct_score",
    ]



def jude_page_1(state:Resume):
    page=state.get('page','')
    print(page)
    if page=="简历评价" or  page== "模拟面试":
        return 'do_evaluate_resume_ready'
    elif page=='简历优化':
        return 'optimize_resume'
    else:
        return 'get_interview_question'

# FIX 3: Changed return value from 'code_test' to 'code_test_preview' to match the node name
def jude_page_2(state:Resume)->Literal['create_radar',END,'code_test_preview']:
    page=state.get('page','')
    if page=="简历评价":
        return 'create_radar'
    elif page=='模拟面试':
        resume_struct_score = state['resume_struct_score']
        experience_score = state['experience_score']
        school_score = state['school_score']
        potential_score = state['potential_score']
        technology_stack_score = state['technology_stack_score']
        if school_score < 60:
            print('很抱歉的通知您，您并未通过我公司的简历筛选。')
            return  END
        final_score = resume_struct_score * 0.1 + experience_score * 0.35 + school_score * 0.15 + potential_score * 0.1 + technology_stack_score * 0.3
        print(final_score)
        if final_score < 75:
            print('很抱歉的通知您，您并未通过我公司的简历筛选。')
            return END
        else:
            print('恭喜你成功通过简历初筛,即将进入算法笔试测试')
            return 'code_test_preview' # <-- FIX 3 HERE

def get_code_question(resume:Resume)->dict:
    global  code
    id=resume.get('code_id','')
    code=Codetest(account=id)
    job=resume['job']
    print('正在获取题目请稍等')
    question=code.get_question(position=job)
    for i,q in enumerate(question,1):
        link=q['link']
        print(f'题目{i}.{link}')
    return {'page':'笔试测试'}

def is_pass_written_examination(state:Resume)->Literal['get_one_interview_question_num',END]:
    global code
    print(code.account)
    print('计时三十分钟.......')
    time.sleep(45*60)
    is_pass=code.is_success()
    if is_pass:
        return 'get_one_interview_question_num'
    print('笔试未通过，抱歉')
    return END

def is_pass_interview(state:Resume)->Literal['get_tow_interview_question_num',END]:
    score = state['answer_score']
    sum_score = score[0] * 0.6 + score[1] * 0.2 + score[2] * 0.2
    if state['page']=='一面':
        if sum_score >=75:
            return 'get_tow_interview_question_num'
        else:
            return END
    elif state['page']=='二面':
        if sum_score >= 75:
            return END
        else:
            return END
    return END

def get_tow_interview_question_num(resume:Resume)->dict:
    return {'interview_question_num':[0,0,1,4],'page':'二面'}

def wait_for_parallel(state: Resume):
    """A simple node to act as a join point after parallel evaluations."""
    return {}
def do_evaluate_resume_ready(state: Resume):
    """A simple node to act as a join point after parallel evaluations."""
    return {}




parallel_nodes = [
        "get_potential_score",
        "get_school_score",
        "get_resume_key_score",
        "get_experience_match_score",
        "get_program_struct_score",
    ]
agent=StateGraph(Resume)
agent.add_node('get_user_resume_message',resume_analyse.analyze_resume)
agent.add_node("get_potential_score", get_potential_score)
agent.add_node("get_school_score", get_school_score)
agent.add_node("get_resume_key_score", get_resume_key_score)
agent.add_node("get_experience_match_score", get_experience_match_score)
agent.add_node("get_program_struct_score", get_program_struct_score)
agent.add_node('do_evaluate_resume_ready',do_evaluate_resume_ready)
agent.add_node('wait_for_parallel', wait_for_parallel)
agent.add_node('create_radar',resume_analyse.create_resume_radar)
agent.add_node('optimize_resume',resume_analyse.optimize_resume)
agent.add_node('create_resume_assessment_report',generate_doc.create_resume_assessment_report)
agent.add_node('code_test_preview',get_code_question)
agent.add_node('get_one_interview_question_num',get_question_num)
agent.add_node('get_interview_question',get_interview_question)
agent.add_node('get_question_answer',get_question_answer)
agent.add_node('get_answer_eval',get_answer_eval)
agent.add_node('generate_question_analyse_report',create_interview_question_analyse_report)
agent.add_node('get_tow_interview_question_num',get_tow_interview_question_num)
agent.add_node('do_answer',do_answer)

# --- 修复从这里开始 ---

# 1. 定义图的入口和基本边
agent.add_edge(START,'get_user_resume_message')
agent.add_edge('optimize_resume',END)

# 2. 添加第一个条件边，并提供 path_map
agent.add_conditional_edges(
    'get_user_resume_message',
    jude_page_1,
    path_map={
        "do_evaluate_resume_ready": "do_evaluate_resume_ready",
        "get_interview_question": "get_interview_question",
        'optimize_resume':'optimize_resume',
        'code_test_preview':'code_test_preview'
    }
)

# 3. 定义并行节点分支
#    'do_evaluate_resume_ready' 节点作为并行分支的起点
for node in parallel_nodes:
    agent.add_edge('do_evaluate_resume_ready', node)
    agent.add_edge(node, 'wait_for_parallel')

# 4. 添加并行结束后的条件边，并提供 path_map
#    'wait_for_parallel' 节点作为并行分支的汇合点
agent.add_conditional_edges(
    'wait_for_parallel',
    jude_page_2,
    path_map={
        "create_radar": "create_radar",
        "code_test_preview": "code_test_preview",
        END: END
    }
)

# 5. 定义 "简历评价" 分支的后续流程
agent.add_edge('create_radar','create_resume_assessment_report')
agent.add_edge('create_resume_assessment_report',END)

# 6. 添加笔试环节的条件边，并提供 path_map
agent.add_conditional_edges(
    'code_test_preview',
    is_pass_written_examination,
    path_map={
        "get_one_interview_question_num": "get_one_interview_question_num",
        END: END
    }
)

# 7. 定义面试的主要流程（这是一个循环体）
agent.add_edge('get_one_interview_question_num','get_interview_question')
agent.add_edge('get_interview_question','do_answer')
agent.add_edge('do_answer','get_question_answer')
agent.add_edge('get_question_answer','get_answer_eval')
agent.add_edge('get_answer_eval','generate_question_analyse_report')

# 8. 添加面试通过与否的条件边，并提供 path_map
agent.add_conditional_edges(
    'generate_question_analyse_report',
    is_pass_interview,
    path_map={
        "get_tow_interview_question_num": "get_tow_interview_question_num",
        END: END
    }
)

# 9. 定义二面流程（回到提问环节，形成循环）
agent.add_edge('get_tow_interview_question_num','get_interview_question')


checkpointer = InMemorySaver()
# 注意：interrupt_before 列表中的节点名必须与 add_node 中的完全一致
agent=agent.compile(checkpointer=checkpointer,interrupt_before=['optimize_resume'])







