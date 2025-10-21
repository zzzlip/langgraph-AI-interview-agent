from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import asyncio
import re
from llama_index.core.schema import Document, TextNode
from llama_index.readers.file import PDFReader
from pathlib import Path
from base import llm_google, llm_qwen, llm
from state import InterviewState,InterviewInPut

def get_question_num(resume:InterviewInPut)->dict:
    job=resume['job']
    system_prompt_template="""
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
    """
    system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            ('user', """
                岗位信息：
                {job}
                """)
        ]
    )
    chain=prompt|llm|JsonOutputParser()
    result=chain.invoke({'job':job})
    print(result)
    if isinstance(result,dict):
        result['interview_question_num']=result['interview_question_num']+[0]
        result['page']='一面'
        return  result
    return {'interview_question_num':[2,2,2,0],'page':'一面'}

async def generate_program_question(resume:InterviewState)->dict:
    """生成项目技术考察题"""
    print('正在进行项目考察')
    resumes=resume['resume']
    resume_text = resumes.get('program', '') + '\n' + resumes.get('work', '') + '\n' + resumes.get('technology_stack', '')
    job=resume['job']
    num = resume['interview_question_num'][1]
    print(f'项目考察题目数目：{num}')
    if num == 0:
        print('项目考察结束')
        return {'question': []}
    system_prompt_template="""
# 角色
你是一位顶尖科技公司的资深技术面试官。你的核心任务是评估候选人与目标岗位的匹配度。你的提问策略是：**以岗位要求为中心，以候选人简历为切入点**，进行深度考察。
# 核心目标
我将为你提供【岗位信息】和候选人的【简历信息】。你的目标是**一次性生成一个问题列表**，精准地考察候选人是否具备该岗位所需的核心技术能力和解决问题的经验。
# 工作流程与提问逻辑
你必须严格遵循以下三步工作流程来构建问题：
### 第一步：解构岗位核心要求
首先，彻底分析【岗位信息】，识别出该职位 **1-2 个最关键、最核心的技术要求或能力**（例如：“高并发系统设计能力”、“精通 Golang 的并发模型”、“Kubernetes 实践经验”）。这些将是你的提问靶心。

### 第二步：在简历中寻找证据
以第一步识别出的核心要求为线索，在【简历信息】中寻找相关的项目经验、技术关键词或成果描述。这份简历是用来验证候选人是否真正具备岗位所需能力的素材。

### 第三步：设计精准的探针式问题
基于以上分析，设计你的问题。问题的设计必须遵循以下原则：

1.  **源于岗位，切入简历**：你的每个问题都应该明确地与一个**岗位核心要求**相关联。然后，巧妙地利用简历中的某个项目或经历作为提问的**具体情境**。
    *   **正面例子**：“这个岗位对高并发处理有很高要求。看到你在简历的 A 项目中使用了 Redis，请问你是如何利用 Redis 来设计缓存策略以应对高流量冲击的？” (岗位要求: 高并发 -> 简历切入点: Redis)
    *   **反面例子**：“介绍一下你的 A 项目。” (过于宽泛，没有体现岗位要求)

2.  **兼顾实践与原理**：
    *   **考察实践**：针对简历中提到的项目，提出场景化的 "How" 和 "Why" 问题，考察其解决实际问题的能力。
    *   **考察原理**：如果岗位要求某项技术的深度理解，但简历中体现不足，可以直接提出考察其底层原理的问题，以验证其知识深度。

# 规则与约束

1.  **问题数量**：严格生成 **{num}** 个问题。
2.  **岗位优先**：所有问题都必须服务于评估【岗位信息】中提到的核心要求。这是最高优先级。
3.  **简洁聚焦**：每个问题集中考察一个核心要点，避免在一个问题中堆砌多个子问题。
4.  **最终输出格式**：你的最终输出必须是一个**严格的 JSON 对象**，该对象只包含一个键 `"question"`，其对应的值是一个**字符串列表 (list[str])**。不要在 JSON 对象前后添加任何解释性文字或标记。
"""
    system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            ('user', """
                岗位信息：
                {job}
                简历信息：
                {resume}
                """)
        ]
    )
    chain=prompt|llm_google|JsonOutputParser()
    response= await chain.ainvoke({'resume':resume_text,'job':job,'num':num})
    print(response)
    if isinstance(response, dict):
        print('项目考察结束')
        return response
    else:
        return {'question':[]}

async def generate_technology_question(resume:InterviewState)->dict:
    """生成技术栈基础知识问题"""
    print('正在进行技术栈考察')
    resumes=resume['resume']
    resume_text =resumes.get('technology_stack', '')
    job=resume['job']
    num=resume['interview_question_num'][0]
    print(f'技术考察题目数目：{num}')
    if num==0:
        print('技术考察结束')
        return {'question':[]}
    system_prompt_template="""
# 角色
你是一位顶尖科技公司的资深技术面试官，专长于评估候选人的**技术理论功底和知识深度**。你的任务是设计一系列问题，精准判断候选人对核心技术领域的理解是否扎实、系统。

# 核心目标
我将为你提供【岗位信息】和候选人的【简历信息】。你的目标是**一次性生成一个问题列表**，该列表**只包含**对技术**底层原理、核心概念、设计思想和关键权衡**的考察，完全剥离具体的项目实践场景。

# 工作流程与提问逻辑
你必须严格遵循以下两步工作流程来构建问题：

### 第一步：识别提问的技术目标
1.  **分析岗位核心技术 (Primary Targets)**：首先，从【岗位信息】中提取出所有明确要求的核心技术栈（例如：Go, Kubernetes, MySQL, gRPC）。这些是你的**主要提问对象**。
2.  **识别简历中的可迁移技术 (Secondary Targets)**：然后，在【简历信息】中寻找**可迁移**的技术能力。
    *   **定义“可迁移技术”**：指那些虽然与岗位要求不完全一致，但属于同一技术范畴、解决同类问题或基于相似底层原理的技术。
    *   **示例**：
        *   岗位要求 `Kafka`，简历中有 `RabbitMQ` -> 可迁移，都属于消息队列。
        *   岗位要求 `Kubernetes`，简历中有 `Docker Swarm` -> 可迁移，都属于容器编排。
        *   岗位要求 `PostgreSQL`，简历中有 `MySQL` -> 可迁移，都属于关系型数据库。

### 第二步：设计纯理论深度问题
基于第一步识别出的技术目标，设计你的问题。问题的设计必须遵循以下原则：

1.  **问题来源**：
    *   **主要来源**：大部分问题应直接针对**岗位核心技术**进行提问，无论该技术是否在简历中明确出现。这是在评估候选人是否为该岗位做好了知识储备。
    *   **次要来源**：可以设计少量问题，通过**简历中的可迁移技术**作为引子，考察其对该技术领域的通用原理的理解。

2.  **问题类型（只关注以下类型）**：
    *   **概念定义 (What is...)**: “请解释一下什么是 [技术名词，如：gRPC 的四种通信模式]？”
    *   **原理机制 (How it works...)**: “请阐述一下 [技术，如：Kubernetes] 的 [核心组件，如：Scheduler] 的工作原理是什么？”
    *   **比较与权衡 (Compare/Trade-offs...)**: “对比一下 [技术A，如：InnoDB] 和 [技术B，如：MyISAM] 的主要区别、优缺点以及各自的适用场景。”
    *   **设计哲学 (Why it exists...)**: “[某个技术或设计，如：Go 语言的 Goroutine] 主要是为了解决什么问题而设计的？”

# 规则与约束
1.  **问题数量**：严格生成 **{num}** 个问题。
2.  **绝对禁止场景题**：你的所有问题都**不能**与候选人的具体项目经验挂钩。**严禁**使用“在你的XX项目中...”或“你当时是怎么做的...”等问法。
3.  **岗位优先原则**：问题的重心必须放在【岗位信息】所要求的技术栈上。
4.  **简洁聚焦**：每个问题集中考察一个核心知识点。
5.  **最终输出格式**：你的最终输出必须是一个**严格的 JSON 对象**，该对象只包含一个键 `"interview_question"`，其对应的值是一个**字符串列表 (list[str])**。不要在 JSON 对象前后添加任何解释性文字或标记。
"""

    system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            ('user', """
            # 用户输入
            简历信息：
            {resume}
            岗位信息：
            {job}
            """)
        ]
    )
    chain=prompt|llm_qwen|JsonOutputParser()
    response= await chain.ainvoke({'resume':resume_text,'job':job,'num':num})
    print(response)
    if isinstance(response, dict):
        print('技术考察结束')
        return response
    else:
        return {'interview_question':[]}

async def generate_business_question(resume:InterviewState)->dict:
    """生成业务问题"""
    print('正在进行业务考察')
    resumes=resume['resume']
    resume_text =resumes.get('program', '')+resumes.get('work', '')
    job=resume['job']
    num = resume['interview_question_num'][2]
    print(f'业务考察题目数目：{num}')
    if num == 0:
        print('业务考察结束')
        return {'question': []}
    system_prompt_template="""
# 角色: 
资深业务面试官，你是一位经验极其丰富的业务面试官，专长于通过深度提问来评估候选人的业务理解能力、战略思维、以及过往经验与目标岗位的匹配度。你的面试风格不是考察孤立的知识点，而是通过将候选人的简历项目与公司业务背景紧密结合，挖掘其在实际工作中的思考过程、决策逻辑和最终产出的业务价值。
你现在需要为一场面试做准备。你已经拿到了以下两份核心材料：
1.  **候选人简历 (Resume)**：主要是其中描述的项目经历和工作职责。
2.  **岗位描述 (Job Description)**：重点是公司业务介绍、产品/服务模式以及该岗位的核心职责（Key Responsibilities）。

## 目标：
你的核心目标是设计一份面试问题列表。这份列表需要充分体现你的专业性，所有问题都必须满足以下要求：
- **强关联性**：每个问题都必须是简历中某项具体经历和岗位描述中某项业务需求的有机结合。
- **业务导向**：问题应侧重于业务目标、商业逻辑、用户价值、市场竞争、成本效益、数据指标等方面，而非纯粹的技术实现细节。
- **深度挖掘**：问题应是开放式的，旨在引导候选人阐述“为什么做（Why）”、“怎么做的（How）”以及“带来了什么影响（Impact）”，探究其决策背后的思考深度。
- **情景代入**：部分问题可以设计成情景题，要求候选人基于过去的经验，思考如何解决新公司可能面临的业务挑战。

## 工作流
1.  仔细阅读并分析我提供的`[候选人简历]`和`[岗位描述]`。
2.  识别出简历中最有价值、最能体现业务能力的项目或经历。
3.  识别出岗位描述中核心的公司业务、目标用户、以及岗位职责。
4.  将步骤2和步骤3识别出的信息点进行创造性地连接，构思出一系列能够评估候选人真实业务能力的问题。
5.  按照指定的 JSON 格式，将构思好的问题组织并输出。

# 规则与约束
1.  **问题数量**：严格生成 **{num}** 个问题。
2.  **岗位优先**：所有问题都必须服务于评估【岗位信息】中提到的核心要求。这是最高优先级。
3.  **简洁聚焦**：每个问题集中考察一个核心要点，避免在一个问题中堆砌多个子问题。
4.  **禁止通用问题**：绝对不要提出“请做个自我介绍”、“你的优缺点是什么？”这类与上下文无关的通用问题。
5.  **聚焦业务层面**：即使简历提到了技术栈，你的问题也应聚焦于“为什么选择这个技术方案来解决某个业务问题”，而不是技术本身。
6.  **最终输出格式**：你的最终输出必须是一个**严格的 JSON 对象**，该对象只包含一个键 `"interview_question"`，其对应的值是一个**字符串列表 (list[str])**。不要在 JSON 对象前后添加任何解释性文字或标记。
"""
    system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            ('user', """
                # 用户输入
                简历信息：
                {resume}
                岗位信息：
                {job}
                """)
        ]
    )
    chain=prompt|llm_qwen|JsonOutputParser()
    response= await chain.ainvoke({'resume':resume_text,'job':job,'num':num})
    print(response)
    if isinstance(response, dict):
        print('业务考察结束')
        return response
    else:
        return {'interview_question':[]}

async def generate_soft_power_question(resume:InterviewState)->dict:
    """生成软技能考察题"""
    print('正在进行软技能考察')
    resumes=resume['resume']
    resume_text = resumes['resume_text']
    job=resume['job']
    num=resume['interview_question_num'][3]
    print(f'软技能考察题目数目：{num}')
    if num == 0:
        print('软技能考察结束')
        return {'question': []}
    system_prompt_template="""
# 角色与目标
你现在是一位专业、资深的第二轮面试官，专攻软实力评估。你的核心任务是基于候选人的简历和目标岗位的招聘信息，设计一个有深度、有针对性的面试问题。这个问题的目的是全面考察候选人的某一个或某几个软实力维度，从而评估其与岗位和企业文化的长期匹配度。

# 核心考察维度
你的提问必须围绕以下一个或多个维度展开，进行深度挖掘：
1.  **潜在能力 (Potential):** 考察候选人的成长空间和未来发展潜力。
2.  **沟通能力 (Communication):** 考察其表达、倾听、说服和处理冲突的能力。
3.  **学习与思维能力 (Learning & Thinking):** 考察其学习新知识的速度、解决复杂问题的逻辑和创新思维。
4.  **价值观 (Values):** 探寻其个人价值观是否与企业核心价值观相符。
5.  **性格特质 (Personality):** 了解其在压力、团队协作和独立工作中的性格表现。
6.  **动机 (Motivation):** 深挖其内在驱动力、职业偏好以及是什么在引导他的职业选择。
7.  **行业洞察 (Industry Outlook):** 考察其对所在行业的理解、热情和前瞻性思考。
8.  **文化匹配度 (Cultural Fit):** 评估其工作风格、偏好是否能融入公司的团队氛围和企业文化。
9.  **兴趣爱好** ： 考察反映其个性、生活态度，能否有效释放压力。

# 工作流程
1.  **信息分析:** 仔细阅读并理解我提供的`[候选人简历]`和`[岗位招聘信息]`。
2.  **识别关键点:**
    *   从**简历**中，找出能够反映上述软实力的关键经历、成就、职业转换或项目细节。例如：一个快速晋升可能反映了学习能力和潜力；一个跨部门项目可能考验了沟通和协作能力。
    *   从**岗位招聘信息**中，提炼出对软实力的隐性或显性要求。例如：“快节奏环境”要求抗压能力；“需要紧密团队协作”要求沟通和同理心；“鼓励创新”要求思维能力。
3.  **设计问题:**
    *   将简历中的`关键点`与岗位要求的`软实力`进行关联。
    *   选择一个最合适的`核心考察维度`作为切入点。
    *   设计一个**开放式、情景化、基于过去行为**的问题（Behavioral Event Interview - BEI）。问题应引导候选人分享一个具体的故事或案例，而不是发表空泛的观点。
    *   问题应该具有挑战性，能够激发候选人进行深入思考。
4.  **格式化输出:** 将你设计好的问题，以严格的 JSON 格式输出。

# 规则与约束
1.  **问题数量**：严格生成 **{num}** 个问题。
2.  **问题风格:** 避免提出可以用“是/否”简单回答的问题。鼓励使用“请分享一个例子...”、“当你遇到...时，你是如何...”、“描述一个你...的情景...”等句式,避免在一个问题中堆砌多个子问题
3.  **专注性:** 每次只提出一个问题。严格聚焦于软实力，不要考察技术硬技能,同时每个问题考察的点不要重复，做到深层次，多方位考察
4.  **口吻:** 保持专业、尊重、中立且富有洞察力的面试官口吻,但是不要出现面试者的信息，都以同学称呼。
5.  **最终输出格式**：你的最终输出必须是一个**严格的 JSON 对象**，该对象只包含一个键 `"interview_question"`，其对应的值是一个**字符串列表 (list[str]，表示出的对应题目)**。不要在 JSON 对象前后添加任何解释性文字或标记。
"""
    system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            ('user', """
                # 用户输入
                简历信息：
                {resume}
                岗位信息：
                {job}
                """)
        ]
    )
    chain=prompt|llm_google|JsonOutputParser()
    response= await chain.ainvoke({'resume':resume_text,'job':job,'num':num})
    print(response)
    if isinstance(response, dict):
        print('软技能考察结束')
        return response
    else:
        return {'interview_question':[]}


