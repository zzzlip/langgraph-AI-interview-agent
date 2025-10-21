
from langgraph.constants import START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langgraph.graph import StateGraph
from typing import Literal

from code_test import create_CodeTest_agent
from resume_analyse import create_resume_evaluate_agent, analyze_resume
from state import  MainState
from interview_agent import create_interview_agent
from resume_optimize import optimize_resume


code_test_agent = create_CodeTest_agent()
resume_evaluate_agent = create_resume_evaluate_agent()
interview_subagent=create_interview_agent()



def rounter_1(state:MainState)->Literal['resume_evaluate_subgraph','resume_optimize','interview_subgraph']:
    """
    用于根据用户选择的模式跳转对应的智能体
    """

    page=state.get('workflow_step','')
    print(page)
    if page=="简历评价" or  page== "模拟面试":
        return 'resume_evaluate_subgraph'
    elif page=='简历优化':
        return 'resume_optimize'
    else:
        return 'interview_subgraph'
def rounter_2(state:MainState)->Literal[END,'code_test_subgraph']:
    """
    用于根据用户选择的模式以及通过程度跳转智能体
    """

    step=state.get('workflow_step','')
    page=state.get('page','')
    if step=="简历评价":
        return END
    elif page==False:
        return END
    else:
        return 'code_test_subgraph'

def rounter_3(state:MainState)->Command:
    """
    用于指引一面结束之后的智能体走向
    """

    step=state.get('workflow_step','')
    page=state.get('page','')
    if step=="模拟面试":
        return Command(
            goto=END
        )
    elif page==False:
        return Command(
            goto=END
        )
    else:
        return Command(
            goto='interview_subgraph',
            update={'interview_question_num':[0,0,3,1]}
        )



main_agent=StateGraph(MainState)
main_agent.add_node('get_user_resume_message', analyze_resume)
main_agent.add_node('resume_evaluate_subgraph', resume_evaluate_agent)
main_agent.add_node('resume_optimize',optimize_resume)
main_agent.add_node('code_test_subgraph', code_test_agent)
main_agent.add_node('interview_subgraph', interview_subagent)

# --- 修复从这里开始 ---

# 1. 定义图的入口和基本边
main_agent.add_edge(START, 'get_user_resume_message')
main_agent.add_conditional_edges(
    'get_user_resume_message',
    rounter_1
)
main_agent.add_conditional_edges(
    'resume_evaluate_subgraph',
    rounter_2
)
main_agent.add_edge('resume_optimize', END)
main_agent.add_conditional_edges(
    'code_test_subgraph',
    lambda x: END if x['page']==False else 'interview_subgraph',
    {END:END,'interview_subgraph':'interview_subgraph'}
)

main_agent.add_conditional_edges(
    'interview_subgraph',
    rounter_3,
{END:END,'interview_subgraph':'interview_subgraph'}
)

checkpointer = InMemorySaver()
# 注意：interrupt_before 列表中的节点名必须与 add_node 中的完全一致
main_agent=main_agent.compile(checkpointer=checkpointer, interrupt_before=['resume_optimize'])


# 保存为PNG格式
try:
    # 需要安装 graphviz 和 pygraphviz
    # pip install graphviz pygraphviz
    png_data = main_agent.get_graph().draw_mermaid_png()
    with open("../图片/main_agent_graph.png", "wb") as f:
        f.write(png_data)
    print("流程图已保存为 workflow_graph.png")
except Exception as e:
    print(f"保存PNG时出错: {e}")
    # 如果无法保存为PNG，至少保存mermaid代码到文件
    with open("workflow_graph.mmd", "w", encoding="utf-8") as f:
        f.write(png_data)
    print("流程图Mermaid代码已保存为 workflow_graph.mmd")







