import json

import jieba
from keybert import KeyBERT
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer, util
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import patheffects
import asyncio
from llama_index.core.schema import TextNode
from llama_index.readers.file import PDFReader
from pathlib import Path
from typing import Annotated
import api_key
from langchain_community.utilities import GoogleSerperAPIWrapper
from base import Resume,llm_google,llm,llm_qwen
import os
import re
from generate_doc import generate_resume_pdf

os.environ["SERPER_API_KEY"] = api_key.search_api_key
encode_path = r"" #sentenc-transformer 模型  我使用的是 shibing624text2vec-base-chinese
STOPWORDS_FILE_PATH = r"stopwords-mast/stopwords-master/scu_stopwords.txt"
model_for_keybert = SentenceTransformer(
    encode_path,
    backend="onnx",
    device="cuda",
    model_kwargs={
        "file_name": "model_O4.onnx",
        "provider": "CUDAExecutionProvider"
    }
)
kw_model = KeyBERT(model=model_for_keybert)

@tool(description="调用搜索引擎")
def web_search(query: Annotated[str, "输入要搜索的内容"]) -> str:
    searches = GoogleSerperAPIWrapper()
    return searches.run(query)


def optimize_resume(resume: Resume) -> dict:
    print('正在进行简历优化')
    resumes = resume['resume_text']
    job = resume.get('job', '')
    path=resume['picture_path']
    prompt_template = """
你现在是一位拥有15年经验的顶尖科技公司（如 Google, Amazon, Meta）的资深技术招聘官，同时也是一位专业的简历优化专家。你非常清楚什么样的简历能在海量简历中脱颖而出，尤其擅长将工程师的技术能力和项目价值通过精炼的语言展现出来。
我是一位技术从业者，正在寻求职业发展机会,我希望你能基于我提供的原始简历，对内容进行深度优化，使其更具吸引力和专业性，同时突出我的核心竞争力。
# Task: 优化简历内容
你的核心任务是分析并优化我提供的简历内容。请严格按照以下要求执行：
1.  **优化【工作经历】和【项目经验】的描述结构**:
    *   将每一条经历都严格按照 **STAR 法则 (Situation, Task, Action, Result)** 的逻辑进行重写。
    *   **背景**: 简要描述项目或工作的背景。
    *   **任务**: 说明你在此情境下需要完成的具体任务或面临的挑战。
    *   **行动**: 详细阐述你为完成任务所采取的具体行动、使用的技术和解决方法。这是描述的重点。
    *   **结果**: 使用**量化**的、有说服力的数据来展示你的行动所带来的成果。例如：“将页面加载速度提升了30%”、“错误率降低了50%”、“新用户转化率提高了15%”等。

2.  **优化内容和措辞**:
    *   使用强有力的**行为动词** (Action Verbs) 开头，例如：**主导 (Led)**、**实现 (Implemented)**、**优化 (Optimized)**、**构建 (Architected/Built)**、**重构 (Refactored)**、**提升 (Increased/Improved)** 等。
    *   如果原始信息中缺少具体的量化结果，请根据项目内容，以 `[建议补充：具体提升了多少百分比？或 节省了多少时间？]` 的形式，在相应位置提出补充建议，以引导我思考并完善。

3.  **优化和补充【技术栈】**:
    *   根据【项目经验】中的描述，为每一个项目精准、全面地提炼、补充和优化其对应的【技术栈】。
    *   技术栈的书写应清晰、有条理，例如：`static: React, TypeScript, Webpack | 后端: Node.js, Express, MySQL | 工具: Git, Docker`。

# Constraints: 必须遵守的规则

1.  **【绝对禁止】** 修改简历中的任何部分标题，例如 "教育背景"、"工作经历"、"项目经验"、"专业技能" 等。
2.  **【绝对禁止】** 改变简历中各个部分的先后顺序。
3.  **【核心原则】** 你的工作**仅限于**优化每个标题下的具体内容描述。保持原始的结构和标题不变。
4.  保持简历的原始语言（例如，如果我提供的是中文简历，优化后也必须是中文）。
5.  输出的最终结果应该是**一份完整且可以直接使用**的简历，而不是仅仅列出修改建议。

# Output Format: 输出格式要求
结构清晰: 最终输出的必须是一份完整且可以直接使用的简历文档。
板块分隔: 在每个一级标题（如【教育背景】、【工作经历】、【项目经验】等）的正下方，都必须添加一条横线 (---) 作为分隔符，以增强简历的结构感和可读性。
纯净输出: 除了简历内容本身，绝对不要包含任何额外的解释、问候或说明文字，如“优化后的简历：”或“以下是为您优化的简历：”等。直接开始输出简历内容。
用户输入：
目标岗位
{job}
原始简历
{resume}
优化后的简历：
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm_google | StrOutputParser()
    full_text = chain.invoke({'job': job, 'resume': resumes})
    if path=='':
        resume_path = generate_resume_pdf(content=full_text)
    else:
        resume_path = generate_resume_pdf(content=full_text,image_path=path)
    return {'optimize_resume_path': resume_path}


def load_chinese_stopwords(filepath):
    """加载中文停用词"""
    if not os.path.exists(filepath):
        print(f"错误: 停用词文件 '{filepath}' 不存在。请检查路径是否正确。")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        print(f"成功加载 {len(stopwords)} 个中文停用词从 '{filepath}'。")
        return stopwords
    except Exception as e:
        print(f"加载停用词文件时发生错误: {e}")
        return []

def extract_keybert_keywords(text, kw_model, stopwords_list, top_n=10, diversity=0.5):
    segmented_text = " ".join(jieba.lcut(text))
    keywords_with_scores = kw_model.extract_keywords(
        segmented_text,
        keyphrase_ngram_range=(3, 7),  # 提取3到7个词的短语
        stop_words=stopwords_list,  # 传入中文停用词列表
        top_n=top_n,  # 提取前N个关键词
        use_mmr=True,  # 使用最大边际相关性，增加关键词多样性
        diversity=diversity  # 多样性参数
    )
    return [kw[0] for kw in keywords_with_scores]


def analyze_resume(resume: Resume) -> dict:
    """
    分析简历PDF文件，提取结构化信息并返回TextNode列表和Resume字典
    参数:
        resume: 包含'path'键的字典，指定PDF文件路径
    返回:
        Resume: 结构化简历数据
    """
    job=resume['job']
    path = resume.get('path', '')
    headers = {
        "教育经历": "school",
        "项目经历": "program",
        "工作经历": "work",
        "实习经历": "work",
        "专业技能": "technology_stack",
        "技能栈": "technology_stack",
        "荣誉奖项": "awards",
        "个人优势": "technology_stack",
        "获奖情况": "awards",
        "志愿者服务经历":"school_work",
        "社团/组织经历": "school_work",
    }
    resume_data = {
        "resume_text": "",
        "program": "",
        "work": "",
        "school": "",
        "technology_stack": "",
        "awards": "",
        "school_work":""
    }

    loader = PDFReader()
    file_path = Path(path)
    documents = loader.load_data(file=file_path)
    nodes = []
    full_text = "\n".join([doc.get_content() for doc in documents])
    resume_data["resume_text"] = full_text
    base_metadata = documents[0].metadata if documents else {}
    regex_pattern = '|'.join(map(re.escape, headers.keys()))
    header_matches = list(re.finditer(regex_pattern, full_text))

    if not header_matches:
        if full_text.strip():
            nodes.append(TextNode(
                text=full_text.strip(),
                metadata={**base_metadata, "header": "文档全文"}
            ))
        return resume_data

    # 处理标题前的内容
    first_header_start = header_matches[0].start()
    if first_header_start > 0:
        content = full_text[:first_header_start].strip()
        if content:
            nodes.append(TextNode(
                text=content,
                metadata={**base_metadata, "header": "求职基本信息"}
            ))

    # 遍历所有标题并提取内容
    for i, match in enumerate(header_matches):
        header = match.group(0)
        content_start = match.end()
        content_end = header_matches[i + 1].start() if i < len(header_matches) - 1 else len(full_text)
        content_block = full_text[content_start:content_end].strip()

        if content_block:
            # 添加到TextNode
            final_text = f"{header}\n{content_block}"
            nodes.append(TextNode(
                text=final_text,
                metadata={**base_metadata, "header": header}
            ))

            # 根据标题类型填充Resume结构
            field = headers.get(header)
            if field:
                resume_data[field] += f"{content_block}\n"
    # 清理多余的空格和换行
    for key in resume_data.keys():
        resume_data[key] = resume_data[key].strip()
    resume_data['job']=job
    print('简历解析完成')
    return resume_data


async def get_resume_key_score(resume: Resume) -> dict:
    """"将用户简历和应聘岗位进行技能匹配度分析并返回匹配分数（满分100分）"""
    print("正在进行简历技术栈匹配分析")
    resume_text = resume.get('program', '') + '\n' + resume.get('work', '') + '\n' + resume.get('technology_stack', '')
    job_description_text = resume.get('job', '')
    prompt_template = """
# 角色
你是一个专业的简历评估专家和职业发展顾问。你的任务是通过深度分析岗位技能需求，对用户简历的技术栈进行一次**结构化、可视化、极具洞察力**的专业评价，并提供 actionable 的建议。

# 核心任务
你的回答必须包含以下六个方面的完整分析，特别是核心技术栈的对比分析需要以一个**高度结构化的诊断表格**呈现。
---

## 1. 总体评价与匹配度

*   **总体符合度：** [高/中高/中/中低/低]
*   **核心亮点：** [简要概述候选人技术栈与岗位要求高度匹配的几个关键点。]
*   **主要差距：** [简要指出候选人技术栈与岗位要求存在的主要差距或不足。]

## 2. 核心技术栈诊断表 (Detailed Diagnosis Table)

请使用以下**高度结构化的表格**，对岗位要求与候选人技术栈进行深度匹配分析。

**图例说明:**
*   **匹配度 ✅:** 高度匹配，是明确的优势项。
*   **匹配度 🟡:** 中等匹配，基本满足要求但有提升空间，或掌握了相关但非首选的技术。
*   **匹配度 ❌:** 存在差距，是明显的短板或技能缺失。

| 技术领域 (Tech Domain) | 岗位具体要求 (Specific Requirement) | 简历体现 (Resume Evidence) | 匹配度 (Match) | 评价与分析 (Evaluation & Analysis) | 提升建议 (Actionable Suggestion) |
|---|---|---|---|
| **后端开发** | 高并发微服务架构 (Go/Java) | 多个项目使用Go和Gin框架，提及QPS优化经验。 | ✅ | **优势项。** 候选人的Go技术栈与岗位要求高度契合，且具备高并发实践经验，这是核心竞争力。 | 可以在面试中准备一个能体现架构设计深度的项目案例。 |
| **前端开发** | 精通React，熟悉其生态 (Redux/Hooks) | 掌握Vue.js，有多个Vue项目经验，未提及React。 | 🟡 | **存在框架差异。** 虽然前端基础扎实，但与岗位首选的React技术栈不符。 | **高优先级。** 建议立即开始学习React，并完成1-2个使用React Hooks和Redux的个人项目。 |
| **数据库** | MySQL设计与优化，有NoSQL实践经验 | 熟悉MySQL和SQL优化，未提及NoSQL。 | 🟡 | **满足部分要求。** MySQL经验符合要求，但缺乏NoSQL实践是短板，可能无法应对岗位对缓存或大数据场景的需求。 | 学习Redis或MongoDB，并在个人项目中集成，重点理解其适用场景和数据模型。 |
| **DevOps** | 熟悉CI/CD, Docker, Kubernetes(K8s) | 熟悉Docker和Jenkins，未提及K8s。 | ❌ | **存在关键差距。** 掌握Docker是基础，但缺乏K8s经验意味着在现代云原生部署和管理方面存在明显短板。 | 学习Kubernetes核心概念（Pod, Service, Deployment），并尝试用Minikube在本地搭建一个简单的应用。 |


**表格填写说明：**
*   **技术领域：** 对技术进行分类，如后端、static、数据库、DevOps、测试、数据分析等。
*   **岗位具体要求：** 深入解读JD，提炼出对技能的具体要求（例如，要求Python，是用于数据分析还是Web开发？）。
*   **简历体现：** 找到简历中能证明该技能的对应描述。
*   **匹配度：** 使用 ✅, 🟡, ❌ 进行可视化评估。
*   **评价与分析：** **只评价，不给建议。** 简洁说明为什么给出此匹配度，点出优劣势。
*   **提升建议：** **只给建议，不评价。** 给出具体、可操作的提升方法。

## 3. 优势技术栈与亮点

*   [基于诊断表的 ✅ 项，详细展开说明候选人的核心优势及其对岗位的价值。]
*   [指出简历中可能被低估但对岗位有潜在价值的技能或特质（例如，某项技术的底层原理理解、优秀的开源贡献等）。]

## 4. 待提升技术栈与差距分析

*   [基于诊断表的 ❌ 和 🟡 项，明确指出候选人需要弥补的核心技能差距。]
*   [分析这些差距可能对候选人胜任该岗位带来的具体影响，例如“缺乏K8s经验可能会在入职后难以快速参与到项目的部署流程中”。]

## 5. 针对性行动计划 (Action Plan)

1.  **技能提升路线图 (Skill Roadmap):**
    *   **高优先级 (0-1个月):** [针对 ❌ 项，给出最紧急的弥补建议，如：学习Kubernetes基础。]
    *   **中优先级 (1-3个月):** [针对 🟡 项，给出完善建议，如：深入学习NoSQL数据库。]
2.  **简历优化建议 (Resume Polish):**
    *   [如果简历中某些技能未能充分体现，给出如何更好地在简历中突出这些技能的建议。例如：“建议将Go项目中的并发优化经验单独作为一点列出，并用数据说明优化成果。”]
3.  **面试准备策略 (Interview Strategy):**
    *   [针对性地建议候选人如何准备面试。例如：“当被问及React经验时，可以诚实说明目前在学习中，并主动将话题引导到你精通的Vue上，通过对比两者来展现你的前端知识深度和学习能力。”]

## 6. 总结
[重申核心观点，并给出最终的综合评价，总结候选人在此岗位上的潜力和挑战。]
---
**用户输入:**
{messages}"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm_google | StrOutputParser()
    response = await chain.ainvoke(
        {'messages': [HumanMessage(content=resume_text + '\ni应聘岗位信息：\n' + job_description_text)]})
    print(response)
    chinese_stopwords_list = load_chinese_stopwords(STOPWORDS_FILE_PATH)
    if not chinese_stopwords_list:
        print("未加载到任何停用词，程序终止。请检查停用词文件路径和内容。")
        exit()
    match_threshold = 0.70  # 相似度阈值，可以根据实际效果调整
    matched_pairs = []
    total_similarity_score = 0
    matched_count = 0
    key_weight = 0.7
    similarity_weight = 0.3
    top_n = 15
    resume_text=resume.get('technology_stack', '')
    resume_keywords_keybert = extract_keybert_keywords(resume_text, kw_model, chinese_stopwords_list, top_n=top_n,
                                                       diversity=0.4)
    job_keywords_keybert = extract_keybert_keywords(job_description_text, kw_model, chinese_stopwords_list, top_n=top_n,
                                                    diversity=0.4)
    print(f"简历关键词: {resume_keywords_keybert}")
    print(f"岗位关键词: {job_keywords_keybert}")
    model_for_comparison = model_for_keybert
    print("\n--- 关键词语义匹配 ---")
    # 编码关键词
    resume_embeddings_keybert = model_for_comparison.encode(resume_keywords_keybert, convert_to_tensor=True)
    job_embeddings_keybert = model_for_comparison.encode(job_keywords_keybert, convert_to_tensor=True)

    for i, r_kw in enumerate(resume_keywords_keybert):
        best_match_score = 0
        best_match_j_kw = None

        for j, j_kw in enumerate(job_keywords_keybert):
            similarity = util.cos_sim(resume_embeddings_keybert[i], job_embeddings_keybert[j]).item()

            if similarity > best_match_score:
                best_match_score = similarity
                best_match_j_kw = j_kw

        if best_match_score >= match_threshold:
            matched_pairs.append((r_kw, best_match_j_kw, best_match_score))
            total_similarity_score += best_match_score
            matched_count += 1

    print("\n匹配结果 (简历关键词 -> 岗位关键词):")
    for r_kw, j_kw, score in sorted(matched_pairs, key=lambda x: x[2], reverse=True):
        print(f"'{r_kw}' <-> '{j_kw}' (相似度: {score:.2f})")
    overall_match_score = total_similarity_score / matched_count if matched_count > 0 else 0
    print(f"\n总匹配关键词数量: {matched_count}")
    print(f"平均匹配相似度: {overall_match_score:.2f}")
    print("\n--- 整体文档相似度 (使用 KeyBERT 内部模型) ---")
    resume_embedding_doc = model_for_comparison.encode(resume_text, convert_to_tensor=True)
    job_embedding_doc = model_for_comparison.encode(job_description_text, convert_to_tensor=True)
    doc_similarity = util.cos_sim(resume_embedding_doc, job_embedding_doc).item()
    print(f"简历与岗位描述整体相似度: {doc_similarity:.2f}")
    sum_score = doc_similarity * similarity_weight + overall_match_score * key_weight
    print(f"最终得分为：{sum_score}")
    print("简历技术栈匹配分析结束")
    return {'technology_stack_evaluate': response, 'technology_stack_score': sum_score * 100}


async def get_school_score(resum: Resume) -> dict:
    """获取院校评分"""
    print("正在进行院校背景分析")
    school = resum.get('school', '')
    job = resum.get('job', '')
    contents=school + '\n'+'应聘岗位信息：\n' + job
    prompt_template = """
    你是一个学历评估专家，你的任务对用户提供的学历信息，依照岗位要求的学历以及学科要求，进行学历背景评分
    你的工作安排：
    首先借助搜索工具对用户提供的自己学历资料进行充分调研
    当认为已经充分掌握用户提供提供学历的资料后，依照岗位进行专业性评分，生成最终答案
    判断标准：
    如果简历中没有对学历进行要求，那么你就需要对从学校层面进行打分，例如学科实力，学校背景等等
    最终答案格式要求：
    以json的格式输出
    school_score 对应字段类型为 float 表示学历背景与岗位要求的匹配得分（满分100 80-100 表示达到要求，且高于岗位学历要求，60-80 表示达到岗位要求。60以下 表示未达到岗位，属于不及格）
    注意严格按照该格式输出
    """
    agent = create_react_agent(llm, [web_search], prompt=prompt_template)
    response = await agent.ainvoke({'messages': [HumanMessage(content=contents)]})
    try:
        ans=response['messages'][-1]
        parse=JsonOutputParser()
        if isinstance(ans,AIMessage):
            ans=parse.parse(ans.content)
            print(ans)
            print("院校背景分析结束")
            return ans
    except Exception as e:
        print(e)
        return {'school_score': 60}


async def get_potential_score(resume: Resume) -> dict:
    """获取未来潜力评分"""
    print("正在进行未来潜力分析")
    text = resume.get('resume_text', '')
    job = resume.get('job', '')
    prompt_template = """
        你是一名资深的人才分析师和潜力评估专家。你的核心任务是基于提供的简历信息和岗位描述，
        对候选人在该岗位上的胜任潜力进行全面、深入且富有洞察力的评估。你的分析必须超越简单的关键词匹配，充分挖掘即使在简历中没有明确提及，
        但通过其他经历、成就、项目、教育背景或个人特质能够侧面证明其在该岗位上具备强大胜任潜力的方面。
        你需要像一名经验丰富的招聘官或猎头一样，识别可迁移技能、学习能力、解决问题的能力、
        以及与岗位核心要求相关的间接证据。
        你可以从下面几个点进行综合评估
        1.  **核心技能与经验匹配度 (Core Skills & Experience Match):**
            *   【直接匹配】：简历中明确提及的，与岗位要求直接相关的关键技能、工具、行业经验、项目经验和成就。
        2.  **可迁移能力与潜力挖掘 (Transferable Skills & Potential Excavation):**
            *   【技能迁移】：即使岗位要求未直接提及，但从其他工作、项目或个人经历中习得，可高度应用于本岗位的通用或专业技能（例如：跨部门协作、数据分析、复杂系统集成、用户研究、内容创作、市场洞察、流程优化等）
            *   【学习能力与适应性】：从教育背景、职业转型、新领域探索、快速掌握新工具/技术、应对变化等经历中体现出的快速学习能力、适应新环境和新挑战的能力。
            *   【解决问题与创新思维】：简历中体现出的分析问题、提出创新解决方案、优化流程、克服困难的案例。这可能体现在任何领域，不限于岗位要求。
            *   【项目管理与执行力】：即使非项目经理岗位，但从组织活动、推动任务、协调资源、达成目标等经历中体现出的规划、执行和交付能力。
            *   【沟通协作与人际影响力】：从团队合作、跨文化交流、客户沟通、领导小团队、影响力构建等经历中体现出的软技能。
            *   【自我驱动与主动性】：从个人项目、志愿活动、职业发展轨迹、自我提升行为中体现出的强烈进取心、主动承担责任、追求卓越的意愿。

        3.  **间接证据与侧面印证 (Indirect Evidence & Lateral Proof):**
            *   【非传统但有价值的经历】：例如，在非相关领域取得的显著成就（如：组织大型社团活动、成功运营个人博客/社区、在某个爱好领域达到专家水平等）
            *   【跨学科或多元背景】：不同寻常的教育背景组合或工作经历，如何带来独特的视角或解决问题的方法，从而为岗位增值。

    最终答案格式要求：
    以json的格式输出 该json中存在的key为 "potential_score" 
    potential_score 对应字段类型为 float 表示学历背景与岗位要求的匹配得分（满分100 80-100 表示潜力值很大，60-80 表示具有一定潜力。60以下 表示看不出用户的潜力所在）
    注意严格按照该格式输出

    用户输入：
    简历信息：
    {resume}
    职位信息
    {job}
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm_google | JsonOutputParser()
    ans = await chain.ainvoke({'resume': text, 'job': job})
    print(ans)
    if isinstance(ans, dict) :
        print("未来潜力分析结束")
        return ans
    else:
        return {"potential_score": 60}


async def get_program_struct_score(resume: Resume) -> dict:
    "将用户简历中的项目和工作经历描述进行书写结构化程度进行分析并返回结构化分数分数（满分100分）"
    print('正在进行简历结构化分析')
    resumes = resume.get('program', '') + '\n' + resume.get('work', '')
    if resumes != '':
        prompt_template = """
                你是一个专业的HR招聘专家，你的任务是针对用户简历中的项目经历（若存在）以及工作经历（若存在）进行结构方面的**深度评价与打分**。你的评价将基于STAR法则，并力求提供**具体、可操作、有洞察力**的改进建议。
                ---
                **评价核心法则：STAR法则（深度解读与评估维度）**
                以下是你评估每个STAR要素时需要考虑的深度维度：

                1.  **Situation（情境）**
                    *   **核心要点：** 简要描述项目/任务所处的宏观背景、面临的挑战、市场需求或问题，以及其复杂性或规模。确保读者能理解为何这项工作是必要的。
                    *   **深度考量维度：**
                        *   **清晰性与简洁性：** 背景描述是否一目了然，没有冗余信息？
                        *   **问题导向性：** 是否有效铺垫了后续任务和行动所要解决的“问题”？
                        *   **关联性与重要性：** 所描述的情境是否与任务和结果紧密相关？是否突出了这项工作的战略意义或紧迫性？
                2.  **Task（任务）**
                    *   **核心要点：** 明确你作为个人（或团队中的你）所承担的具体任务、目标或职责。强调目标的可衡量性或具体性。
                    *   **深度考量维度：**
                        *   **具体性与可衡量性：** 任务目标是否具体（SMART原则：Specific, Measurable, Achievable, Relevant, Time-bound）？
                        *   **个人职责界定：** 是否清晰地界定了你在团队中的个人职责和贡献，而非泛泛而谈团队目标？
                        *   **挑战性与主动性：** 任务是否体现了一定的挑战性？你是否在任务中展现了主动性？

                3.  **Action（行动）**
                    *   **核心要点：** 详细说明你采取的具体策略、技术方法、工具、决策过程以及你在其中扮演的角色和贡献。重点突出你的思考过程、解决问题的方法和创新点。
                    *   **深度考量维度：**
                        *   **具体性与可复现性：** 行动描述是否足够具体，让读者能想象你“是如何做到的”？是否涵盖了关键步骤和技术细节？
                        *   **个人贡献突出：** 是否清晰地阐述了“你”具体做了什么，而非仅仅罗列团队工作或项目内容？
                        *   **问题解决能力：** 行动是否体现了你分析问题、解决问题的思维过程和能力？是否有对技术选型、方案对比的思考？
                        *   **创新性与影响力：** 你的行动是否有任何创新点？这些行动如何直接促成了结果？

                4.  **Result（结果）**
                    *   **核心要点：** 用量化数据或事实展示行动带来的具体影响、成果和价值。结果应直接对应S和T中的问题与目标，并体现你的贡献。
                    *   **深度考量维度：**
                        *   **量化与可衡量性：** 结果是否用具体数据（数字、百分比、金额、时间等）进行了量化？
                        *   **影响力与价值：** 结果是否体现了对业务、用户、效率、成本或收入的实际影响？是否与S和T中提出的问题或目标直接对应？
                        *   **可信度与相关性：** 结果是否真实可信？是否直接由你所描述的行动促成？

                ---
                **评价流程与输出格式要求：**
                请你根据上述STAR法则的深度考量维度，对用户简历中提供的项目经历和工作经历，按照以下格式输出。
               请对用户提供的每一段项目/工作经历，生成独立的诊断报告。报告的主体内容必须使用**Markdown表格**来呈现，以确保清晰和直观。

                ### 评价项目/工作经历：[项目/工作经历名称]
                
                #### 1. STAR法则深度诊断表
                *使用一个Markdown表格，对STAR的四个要素进行逐一诊断。*
                
                | STAR 要素 (Element) | 现状分析 (Current State Analysis) | 改进建议 (Improvement Suggestion) | 要素得分 (/25) |
                |---|---|---|---|
                | **S (情境)** | [此处分析简历原文中S部分的优缺点。例如：优点是点明了项目背景，缺点是未能突出挑战的复杂性。] | [此处提供具体、可操作的修改建议。例如：建议增加一句关于当时市场竞争激烈或技术瓶颈的描述，以凸显项目的必要性。] | [为该要素打分] |
                | **T (任务)** | [此处分析T部分的优缺点。例如：清晰地说明了个人负责的模块，但目标不够具体，缺乏可衡量的KPI。] | [此处提供具体建议。例如：建议将“负责优化性能”改为“目标是将接口P99延迟从500ms降低到200ms以内”。] | [为该要素打分] |
                | **A (行动)** | [此处分析A部分的优缺点。例如：罗列了使用的技术栈，但缺乏个人决策过程和解决问题的细节。] | [此处提供具体建议。例如：建议补充说明“为什么选择Redis而非Memcached”，并描述你是如何设计缓存更新策略来解决数据一致性问题的。] | [为该要素打分] |
                | **R (结果)** | [此处分析R部分的优缺点。例如：提到了“性能得到提升”，但描述过于模糊，缺乏说服力。] | [此处提供具体建议。例如：强烈建议量化成果，如“最终接口P99延迟降低至180ms，系统吞吐量提升了150%”。] | [为该要素打分] |

                #### 2. 总体评价与核心建议
                [在表格下方，对该段经历进行简要的总体评价。指出最主要的亮点和最需要优先改进的部分。例如：整体来看，该项目经历展现了你的技术执行力，但价值呈现（Result）和思考深度（Action）是当前的主要短板。建议你优先将成果量化，并补充技术决策的思考过程。]
                    项目经历：
                    {program}
                    输出格式：
                    要求以json的格式输出 使用的键名为 "resume_struct_evaluate"," resume_struct_score "
                    resume_struct_evaluate 字段对应类型为str 表示为对简历工作或者项目经历的评价
                    resume_struct_score 字段类型为float 表示对历工作或者项目经历书写的总体评分（满分100,80-100 表示书写的结构完全符合STAR结构，只是可能有一点瑕疵。 60-80 表示基本符合STAR结构，但是四元素中的某几部分不符合。60以下 表示基本不符合STAR结构，结构混乱，属于不及格
                    """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm_qwen | JsonOutputParser()
        response=await chain.ainvoke({'program': resumes})
        print(response)
        if isinstance(response, dict) :
            print('简历结构化分析结束')
            return response

        else:
            return {'resume_struct_evaluate': '', 'resume_struct_score': 60.0}


async def get_experience_match_score(resume: Resume) -> dict:
    print("正在进行简历经验匹配分析")
    resumes = resume.get('program', '') + resume.get('work', '') + resume.get('awards', '')
    job = resume.get('job', '')
    prompt_template = """
    # 角色
你是一位顶尖的职业发展顾问和简历分析专家。你的任务是深度分析用户简历与目标岗位的匹配度，并生成一份**结构化、数据驱动、高度可读的JSON格式评估报告**。

# 核心任务
我将为你提供【简历信息】和【目标岗位】。你的任务是：
1.  **前期调研 (Research Phase)**：调用搜索工具，为后续的建议部分收集必要的外部信息。
2.  **生成报告 (Generation Phase)**：将所有分析和建议整合到一个**单一的JSON对象**中。你必须**确保“发展建议”部分详细、具体地整合并阐述你在第一步中调研到的所有信息**。

# 工作流程

### 第一步：前期调研（调用搜索工具）
在生成最终答案之前，你必须先调用搜索工具完成以下调研任务，为“深度发展计划”部分做准备：
1.  **搜索开源项目**：查找并筛选2-3个与【目标岗位】技能要求**高度相关**的开源项目（如GitHub上的项目），**记录下具体的名称、链接及其核心技术特点**。
2.  **搜索竞赛活动**：查找并筛选1-2个与【目标岗位】相关的专业比赛、技术挑战赛或重要的行业会议，**记录下具体名称和官网链接**。
3.  **搜索学习资源与社区**：查找并筛选与【目标岗位】核心技能相关的：
    *   1-2个高质量的在线课程或深度教程。
    *   1-2位行业内的技术专家或其有影响力的技术博客。
    *   1-2个活跃的线上技术社区（如Slack/Discord频道、专业论坛）。

### 第二步：生成JSON评估报告
调研完成后，严格按照#输出格式的要求，生成最终的JSON报告。

#### 1. 详细匹配度分析 (Detailed Match Analysis)
**分析内容**：评估简历中的经验、技能与岗位要求的契合度。
输出格式：
使用一个Markdown表格，进行评估
|---|---|---|---|
| 评估项 (Item) | 匹配状态 (Status) | 详细分析 (Analysis) |
| [在此处填充分析内容] | [在此处填充分析内容] | [在此处填充分析内容] |
| ... | ... | ... |

#### 2. 基于调研的深度发展计划 (In-depth Development Plan Based on Research)
*   **分析内容**：**你必须将第一步调研到的所有结果，深度整合并阐述到下方的表格中**。建议你可以借鉴你的调研结果，并提供清晰的实施路径。
输出格式：
*使用Markdown表格来整合所有建议。*
|---|---|---|---|
| 建议类别 (Category) | 具体建议 (Suggestion) | 调研来源/链接 (Source/Link from Research) | 实施路径 (Implementation Path) | 预期收益 (Expected Benefit) |
| **项目实践** | 贡献或复现 `milvus-io/milvus` 项目中的部分功能，特别是其向量索引或数据管理模块。 | `github.com/milvus-io/milvus` | 1. **Fork并搭建**：在本地成功运行Milvus。 <br> 2. **代码研读**：重点阅读`internal/core`目录，理解其核心架构。 <br> 3. **任务切入**：从标记为 `good first issue` 或 `help wanted` 的任务开始，尝试提交一个小的PR。 | 获得业界领先的向量数据库项目经验，这与大模型、搜索推荐等前沿岗位需求高度相关，是简历的巨大加分项。 |
| **竞赛活动** | 参加或复盘“Kaggle”上的“LLM - Science Exam”等自然语言处理相关的竞赛。 | `kaggle.com/competitions` | 1. **赛题理解**：深入分析比赛的评估指标和数据集。 <br> 2. **方案复现**：学习并复现Top 10%选手的开源解决方案。 <br> 3. **创新实践**：在复现基础上，尝试改进模型或特征工程，并记录实验结果。 | 在实践中检验和提升你的模型调优、特征工程和解决复杂问题的能力，获得有说服力的数据成果。 |
| **专业学习与社区参与** | 深入学习`Hugging Face`官方课程，并关注其核心开发者`Thomas Wolf`的技术分享。 | - `huggingface.co/learn/nlp-course` <br> - Twitter: `@Thom_Wolf` | 1. **系统学习**：完成Hugging Face NLP课程，并动手完成所有代码实验。 <br> 2. **主动关注**：在Twitter或博客上关注专家的最新动态，理解前沿技术趋势。 <br> 3. **社区互动**：加入Hugging Face的Discord或论坛，参与技术讨论，尝试回答新手问题。 | 系统性地掌握SOTA（最先进的）NLP工具库，接轨行业标准，并通过社区建立个人技术品牌和影响力。 |


#### 3. 总结与展望 (Summary & Outlook)
*   **分析内容**：对整体评估进行简要总结，并给出鼓励性的展望和下一步行动的核心建议。
*   **呈现方式**：使用常规的Markdown文本格式。

---

# 输出格式
你的最终输出**必须**是一个严格的、没有任何多余字符的JSON格式，其中包含两个键：`experience_evaluate` 和 `experience_score`。

*   `"experience_evaluate"`: (string)
    *   其值是一个**包含完整评估报告的字符串**。
*   `"experience_score"`: (float)
    *   其值是一个浮点数，表示简历与岗位的匹配度总分（满分100）。
    *   评分标准：80-100分 (高度匹配), 60-80分 (基本匹配), 60分以下 (匹配度较低)。
    """
    agent = create_react_agent(llm, [web_search], prompt=prompt_template)
    response = await agent.ainvoke({'messages':resumes+'应聘岗位信息：\n'+ job})
    try:
        ans=response['messages'][-1]
        parse=JsonOutputParser()
        if isinstance(ans,AIMessage):
            ans=parse.parse(ans.content)
            print(ans)
            print("简历匹配分析结束")
            return ans
    except Exception as e:
        print(e)


def create_resume_radar(resume: Resume) -> dict:
    """
    创建简历评价雷达图（五维度版）
    """
    structural_score = resume.get('resume_struct_score', 0)
    technical_score = resume.get('technology_stack_score', 0)
    school_score = resume.get('school_score', 0)
    experience_score = resume.get('experience_score', 0)
    potential_score = resume.get('potential_score', 0)
    save_path = '雷达图/five_dimension_radar.png'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    labels = ['简历结构', '经历匹配', '技术能力', '学校背景', '发展潜力']
    data = [structural_score, experience_score, technical_score, school_score, potential_score]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    data = np.concatenate((data, [data[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    grid_color = '#e0e0e0'
    text_color = '#333333'
    accent_color = '#2563eb'
    for i in range(1, 6):
        ax.plot(angles, [i * 20] * len(angles), color=grid_color, alpha=0.7, linewidth=1, linestyle='--')
    ax.fill(angles, [100] * len(angles), facecolor='#f5f5f5', alpha=0.3)
    ax.fill(angles, [80] * len(angles), facecolor='#f0f0f0', alpha=0.3)
    ax.fill(angles, [60] * len(angles), facecolor='#ebebeb', alpha=0.3)
    ax.fill(angles, [40] * len(angles), facecolor='#e6e6e6', alpha=0.3)
    ax.fill(angles, [20] * len(angles), facecolor='#e1e1e1', alpha=0.3)
    line, = ax.plot(angles, data, color=accent_color, linewidth=3, marker='o',
                    markersize=10, markerfacecolor=accent_color, markeredgecolor='white')
    line.set_path_effects([
        patheffects.withStroke(linewidth=5, foreground=accent_color, alpha=0.2),
        patheffects.withStroke(linewidth=8, foreground=accent_color, alpha=0.1)
    ])
    ax.fill(angles, data, facecolor=accent_color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=text_color, fontsize=12, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"],
               color=text_color, size=10, fontweight='bold')
    plt.ylim(0, 110)
    for angle, value in zip(angles[:-1], data[:-1]):
        ax.text(angle, value + 7, f"{value}",
                color='white', ha='center', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(facecolor=accent_color, alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))
    center_circle = Circle((0.5, 0.5), 0.05, transform=ax.transAxes,
                           color=accent_color, alpha=0.8, zorder=10)
    fig.add_artist(center_circle)
    title = ax.set_title('简历综合评价雷达图', color=text_color, fontsize=18, fontweight='bold', pad=30)
    title.set_path_effects([
        patheffects.withStroke(linewidth=3, foreground=accent_color, alpha=0.3)
    ])
    for i in range(36):
        angle = np.radians(i * 10)
        ax.plot([angle, angle], [0, 105], color=grid_color, alpha=0.3, linewidth=0.5)
    outer_circle = Circle((0.5, 0.5), 0.45, transform=ax.transAxes,
                          facecolor='none', edgecolor=accent_color, alpha=0.3, linewidth=2)
    fig.add_artist(outer_circle)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"五维评价雷达图已保存至: {save_path}")
    return {'resume_radar_path': save_path}


async def get_resume_five_dimension(resume: Resume) -> dict:
    print('正在进行简历五维度分析')
    result = await asyncio.gather(
        get_potential_score(resume),
        get_school_score(resume),
        get_resume_key_score(resume),
        get_experience_match_score(resume),
        get_program_struct_score(resume)
    )
    print(result)
    answer = {}
    for r in result:
        try:
            for key, value in r.items():
                answer[key] = value
        except Exception as e:
            print(e)
    return answer


