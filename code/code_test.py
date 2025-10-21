import time

import requests
import asyncio
from langchain_core.runnables import RunnableConfig


from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send

from base import llm
from callbacks import llm_callback_handler
from state import CodeTestInput, CodeState, CodeTestOutPut, CodeQuestionOutput

# --- 1. 定义、模型和状态 ---

all_tags = [
    "data structures", "dp", "graphs", "binary search", "greedy", "math",
    "implementation", "strings", "sortings", "divide and conquer",
    "brute force", "trees", "dfs and similar", "two pointers"
]




# --- 2. 定义图的节点  ---
async def get_question_difficult(state: CodeTestInput,config: RunnableConfig) -> dict:
    """获取所要给出算法题的难度评分"""
    print(config)
    print("--- 步骤1: 获取题目难度分类 ---")
    system_prompt_template = """
    你是一位算法面试官，你的任务是根据用户投递的公司和岗位为用户出5道相关的算法题目 
    要求：
    我会给你相关算法的标签 你需要根据职位需求以及以往该公司的出题习惯（若存在），选取相关标签的题目 并给出标签 难度系数，和该标签题目数量
    出题规则：
    1.rating在800-1000 代表简单题目 1000到1300 代表中等题目 1300-1700 代表高难度题目。
    2.整个出题顺序要按照从简单到困难的的格式，同时如果应聘的职位对代码能力要求低或者所应聘公司并非为中大厂，那么题目难度不能太高。
    3.注意所有题目的rating都应该在1700以下。
    输出格式
    {output_instruction}
    所有面试算法题的相关标签汇总
    {all_tag}
    用户应聘的职位信息
    {position}
    """
    prase = JsonOutputParser(pydantic_object=CodeQuestionOutput)
    system_prompt = SystemMessagePromptTemplate.from_template(
        system_prompt_template,
        partial_variables={'output_instruction': prase.get_format_instructions(), 'all_tag': all_tags}
    )
    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        ("user", "用户应聘的职位信息\n{position}")
    ])

    chain = (prompt | llm | prase).with_config({"callbacks": llm_callback_handler})

    position = state['position']
    response = await chain.ainvoke({'position': position})
    return {
        'question_class': response,
        'code_id': state['code_id'],  # 传递 code_id 到状态
        'position': position
    }


async def rounter(state: CodeState):
    """分发任务到 get_code_question 节点"""
    tasks = [
        Send('get_code_question', {'tag': t, 'score': s, 'count': c})
        for t, s, c in
        zip(state['question_class']['tag'], state['question_class']['score'], state['question_class']['count'])
    ]
    print(f"\n--- 步骤2: Router 分发 {len(tasks)} 个并发任务 ---")
    return tasks


async def get_problems_by_rating_and_tag(state: dict) -> dict:
    """根据分数和标签并发地获取题目链接。"""
    rating: int = state['score']
    tag: str = state['tag']
    count: int = state['count']
    print(f"  -> 开始并发查找 - 标签: {tag}, 难度: {rating}, 数量: {count}")

    target_index_letter = await _get_index_from_rating_boundary(rating)
    if not target_index_letter:
        return {'question': []}

    url = "https://codeforces.com/api/problemset.problems"

    # 【关键修正】使用 asyncio.to_thread 运行阻塞的 requests 调用
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=10)
    except Exception as e:
        print(f"  -> HTTP 请求出错 (标签: {tag}): {e}")
        return {"question": []}

    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            problems = data["result"]["problems"]
            problem_links = []
            problem_count = 0
            for problem in problems:
                if problem.get("index") == target_index_letter and tag in problem.get("tags", []):
                    contest_id = problem.get("contestId")
                    index_in_contest = problem.get("index")
                    if contest_id and index_in_contest:
                        problem_links.append({
                            "name": problem["name"],
                            "link": f"https://codeforces.com/problemset/problem/{contest_id}/{index_in_contest}",
                            "contestId": contest_id,
                            "index": index_in_contest,
                            "rating": problem.get("rating"),
                            "tags": problem.get("tags", [])
                        })
                        problem_count += 1
                        if problem_count >= count:
                            break
            print(f"  <- 查找完成 - 标签: {tag}, 找到: {len(problem_links)} 道题")
            return {"question": problem_links}
    print(f"  <- HTTP 错误或API问题 - 标签: {tag}, 状态码: {response.status_code}")
    return {"question": []}


async def _get_index_from_rating_boundary(rating_boundary: int) -> str:
    if rating_boundary is None: return ''
    if rating_boundary < 900:
        return 'A'
    elif 900 <= rating_boundary < 1000:
        return 'B'
    elif 1000 <= rating_boundary < 1200:
        return 'C'
    elif 1200 <= rating_boundary < 1400:
        return 'D'
    elif 1400 <= rating_boundary < 1500:
        return 'E'
    elif 1500 <= rating_boundary < 1600:
        return 'F'
    elif rating_boundary >= 1600:
        return 'G'
    return ''


async def get_user_submissions_for_problem(account: str, contest_id: int, problem_index: str) -> dict:
    """获取单个题目的提交情况（辅助函数）"""
    url = f"https://codeforces.com/api/user.status?handle={account}&from=1&count=100"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "OK":
                for submission in data["result"]:
                    if (submission.get("contestId") == contest_id and
                            submission.get("problem", {}).get("index") == problem_index and
                            submission.get("verdict") == 'OK'):
                        return {"verdict": "OK"}
    except Exception as e:
        print(f"获取用户 {account} 提交时出错: {e}")
    return {"verdict": "WRONG_ANSWER"}  # 默认未通过


async def is_success(state: CodeState) -> dict:
    """判断是否通过笔试"""
    time.sleep(45 * 60)
    print("\n--- 步骤3: 并发检查用户所有题目的提交记录 ---")
    account = state['code_id']
    questions = state['question']
    tasks = [
        get_user_submissions_for_problem(account, q['contestId'], q['index'])
        for q in questions
    ]
    results = await asyncio.gather(*tasks)
    num_ok = sum(1 for res in results if res['verdict'] == "OK")
    print(f"检查完成: 用户 {account} 通过了 {num_ok} / {len(questions)} 道题。")
    if num_ok >= 2:  # 通过标准是至少2题
        return {"page": True}
    return {"page": False}


# --- 3. 构建图 ---

def create_CodeTest_agent():
    graph = StateGraph(CodeState, input_schema=CodeTestInput, output_schema=CodeTestOutPut)

    graph.add_node('get_question_difficulty', get_question_difficult)
    graph.add_node('get_code_question', get_problems_by_rating_and_tag)
    graph.add_node('is_success', is_success)

    graph.add_edge(START, 'get_question_difficulty')

    # 条件边：从难度分类节点分发到题目获取节点
    graph.add_conditional_edges(
        'get_question_difficulty',
        rounter,
        {'get_code_question': 'get_code_question'}
    )

    # 聚合边：所有题目获取任务完成后，进入成功判断节点
    graph.add_edge('get_code_question', 'is_success')
    graph.add_edge('is_success', END)
    agent=graph.compile(interrupt_before=['is_success'])
    return agent

