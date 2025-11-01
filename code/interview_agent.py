import asyncio

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from generate_doc import create_interview_question_analyse_report
from state import InterviewState,InterviewOutPut
from generate_interview_question import  get_question_num,generate_program_question,generate_business_question,generate_technology_question,generate_soft_power_question
from generate_question_eval import  get_answer_eval,get_user_question_answer,do_answer


def create_interview_agent():
    graph=StateGraph(InterviewState,output_schema=InterviewOutPut)
    graph.add_node(get_question_num.__name__,get_question_num)
    graph.add_node(generate_program_question.__name__,generate_program_question)
    graph.add_node(generate_business_question.__name__,generate_business_question)
    graph.add_node(generate_technology_question.__name__,generate_technology_question)
    graph.add_node(generate_soft_power_question.__name__,generate_soft_power_question)
    graph.add_node('user_doing_question',do_answer)
    graph.add_node('get_user_question_answer', get_user_question_answer)
    graph.add_node('get_question_eval',get_answer_eval)
    graph.add_node('generate_question_analyse_report', create_interview_question_analyse_report)

    parrable_node_name=['generate_program_question','generate_business_question','generate_technology_question','generate_soft_power_question']
    graph.add_edge(START,'get_question_num')
    for node in parrable_node_name:
        graph.add_edge('get_question_num',node)
        graph.add_edge(node,'user_doing_question')
    graph.add_edge('user_doing_question','get_user_question_answer')
    graph.add_edge('get_user_question_answer','get_question_eval')
    graph.add_edge('get_question_eval','generate_question_analyse_report')
    graph.add_edge('generate_question_analyse_report',END)
    agent=graph.compile(interrupt_before=['get_user_question_answer'])
    return agent













