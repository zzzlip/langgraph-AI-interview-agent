import requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import api_key
from base import llm_qwen


class Codetest():
    def __init__(self,account:str):
        self.account = account
        self.all_tags =[
    "data structures",
    "dp",
    "graphs",
    "binary search",
    "greedy",
    "math",
    "implementation",
    "strings",
    "sortings",
    "divide and conquer",
    "brute force",
    "trees",
    "dfs and similar",
    "two pointers"
]
        self.llm=llm_qwen
        self.question_link=[]
        self.all_submissions = []

    def _get_index_from_rating_boundary(self, rating_boundary: int) -> str:
        """
        根据分数边界确定对应的题目索引字母 (A, B, C, D, E, F, G)。
        A: rating < 900
        B: 900 <= rating < 1000
        C: 1000 <= rating < 1200
        D: 1300 <= rating < 1400
        E: 1400 <= rating < 1500
        F: 1500 <= rating < 1600
        G: rating >= 1600
        """
        if rating_boundary is None:
            return ''
        if rating_boundary < 900:
            return 'A'
        elif 900 <= rating_boundary < 1000:
            return 'B'
        elif 1000 <= rating_boundary < 1200:
            return 'C'
        elif 1300 <= rating_boundary < 1400:
            return 'D'
        elif 1400 <= rating_boundary < 1500:
            return 'E'
        elif 1500 <= rating_boundary < 1600:
            return 'F'
        elif rating_boundary >= 1600:
            return 'G'
        return '' # 对于不在定义范围内的分数

    def get_problems_by_rating_and_tag(self, rating: int, tag: str, count: int) -> list[dict]:
        """
        根据分数范围（转换为索引字母）和标签返回数目的题目链接。
        """
        # 根据 min_rating 确定目标索引字母
        # 假设 min_rating 会与某个定义类别的起始分数对齐
        target_index_letter = self._get_index_from_rating_boundary(rating)
        if not target_index_letter :
            return []

        url = "https://codeforces.com/api/problemset.problems"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "OK":
                problems = data["result"]["problems"]
                problem_links = []
                problem_count = 0  # 追踪已找到的题目数量
                for problem in problems:
                    # 筛选题目索引字母
                    index = problem.get("index")
                    if index is None or index != target_index_letter:
                        continue

                    # 筛选标签
                    tags = problem.get("tags", [])
                    if tag is not None and tag not in tags:
                        continue
                    contest_id = problem.get("contestId")
                    # problem.get("index") 已经是题目在比赛中的字母索引 (A, B, C...)
                    index_in_contest = problem.get("index")
                    if contest_id and index_in_contest:
                        problem_link = f"https://codeforces.com/problemset/problem/{contest_id}/{index_in_contest}"
                        problem_links.append({
                            "name": problem["name"],
                            "link": problem_link,
                            "contestId": contest_id,
                            "index": index_in_contest,
                            "rating": problem.get("rating"),  # 保留原始 rating 信息
                            "tags": tags
                        })
                        problem_count += 1
                        if count is not None and problem_count >= count:
                            break  # 达到题目数量限制，退出循环

                return problem_links
            else:
                print(f"API error: {data['comment']}")
                return []
        else:
            print(f"HTTP error: {response.status_code}")
            return []

    def get_user_submissions_for_problem(self):
        """获取特定用户在特定题目上的提交情况."""
        url = f"https://codeforces.com/api/user.status?handle={self.account}&from=1&count=500" #调整count以获取足够的提交
        response = requests.get(url)
        print(self.question_link)
        for submission in self.question_link:
            contest_id = submission["contestId"]
            problem_index = submission["index"]
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "OK":
                    submissions = data["result"]
                    relevant_submissions = []
                    for submission1 in submissions:
                      if (submission1.get("contestId") == contest_id and
                        submission1.get("problem", {}).get("index") == problem_index): # nested get
                        relevant_submissions.append(submission1)
                    for id,submission2 in enumerate(relevant_submissions):
                        if submission2.get("verdict") == 'OK':
                            self.all_submissions.append(submission2)
                            break
                        if id==len(relevant_submissions)-1:
                            relevant_submissions[0]['verdict'] ="WRONG_ANSWER"
                            self.all_submissions.append(relevant_submissions[0])
                    else:
                        self.all_submissions.append({'name':submission['name'],'tags':submission['tags'],'verdict':"WRONG_ANSWER"})

            else:
                print(f"HTTP error: {response.status_code}")

    def get_question(self,position:str)->list[dict]:
        """获取所要给出算法题的难度评分，并获得出的题目"""
        prompt_template="""
        你是一位算法面试官，你的任务是根据用户投递的公司和岗位为用户出5道相关的算法题目 
        要求：
        我会给你相关算法的标签 你需要根据职位需求以及以往该公司的出题习惯（若存在），选取相关标签的题目 并给出标签 难度系数，和该标签题目数量
        出题规则：
        1.rating在800-1000 代表简单题目 1000到1300 代表中等题目 1300-1700 代表高难度题目。
        2.整个出题顺序要按照从简单到困难的的格式，同时如果应聘的职位对代码能力要求低或者所应聘公司并非为中大厂，那么题目难度不能太高。
        3.注意所有题目的rating都应该在1700以下。
        你要以json的格式输出
        要求如下：
        1.tag字段对应类型为 list[str] 表示所要抽取的题目对应的标签，每一类只给出一个标签。
        2.score字段对应类型为 list[int] 表示对应标签所出题目的难度分数,每一个标签只用一个数字表示即可
        3.count 字段队对应类型为 list[int] 表示对应标签所出题目个数，一定要注意总体题目数量要等于5。
        所有面试算法题的相关标签汇总
        {all_tag}
        用户应聘的职位信息
        {position}
        """
        prompt=ChatPromptTemplate.from_template(prompt_template)
        llm=prompt|self.llm|JsonOutputParser()
        response=llm.invoke({"all_tag":self.all_tags,"position":position})
        print(response)
        tag=response["tag"]
        rating=response["score"]
        count=response["count"]
        for t,s,c in zip(tag,rating,count):
            self.question_link+=self.get_problems_by_rating_and_tag(rating=s,tag=t,count=c)
        return self.question_link

    def is_success(self)->bool:
        """判断是否通过笔试"""
        num=0
        self.get_user_submissions_for_problem()
        for submission in self.all_submissions:
            if submission['verdict'] == "OK":
                num+=1
        if num>=2:
            return True
        return False