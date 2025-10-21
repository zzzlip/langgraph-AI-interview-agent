import json

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Document,Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
import chromadb
from base import llm
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
Settings.llm=llm
path = r"" #emmbeding模型
rerank_model_name = r""  # 示例模型
# 注意：对于中文，'BAAI/bge-reranker-base' 或 'BAAI/bge-reranker-large' 通常是更好的选择
# rerank_model_name = "BAAI/bge-reranker-base" # 如果处理中文内容，可以尝试这个
embeddings = HuggingFaceEmbedding(model_name=path)
Settings.embed_model = embeddings
job_name=['数据产品经理', '产品运营专家', '后端开发工程师', '算法工程师', '推荐算法工程师', '自然语言处理工程师', '前端开发工程师', '数据工程师', '数据科学家', '云计算工程师', '全栈工程师', '架构师']
company_name=['阿里巴巴', '美团', '腾讯', '华为', '字节跳动', '京东', '微软', '百度']
persist_path= "../chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_or_create_collection(name="interview_collection")
vector_store = ChromaVectorStore(
        chroma_collection=collection,
        collection_name="interview_collection",
    )
def create_chroma_db():
    full_doc=[]
    data=json.load(open("../面试知识库/interview_questions.json", "r", encoding="utf-8"))
    print(len(data['分类列表']))
    for d in data['分类列表']:
        question=d['问题列表']
        for q in question:
            print(q)
            text=q.get("问题内容",'')
            types=q.get('面试类型','')
            companies=q['关联公司']
            jobs=q['关联岗位']
            for company in companies:
                for job in jobs:
                    # 4. 在内层循环中，为每一对 company 和 job 的组合创建一个 Document 对象
                    print(f"正在创建组合: 公司='{company}', 岗位='{job}'")  # 添加打印语句，方便理解过程
                    doc = Document(
                        text=text,
                        metadata={
                            'interview_type': types,
                            'company': company,  # 'company' 来自外层循环
                            'job': job,  # 'job' 来自内层循环
                        },
                        metadata_seperator="::",
                        metadata_template="{key}=>{value}",
                        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
                    )
                    full_doc.append(doc)

    data=json.load(open("../面试知识库/JAVA八股文(1).json", "r", encoding="utf-8"))
    for q in data['技能']:
        text=q.get("描述",'')
        types=q.get('面试类型','')
        job="JAVA工程师"
        for company in company_name:
            for job in job_name:
                doc=Document(
                    text=text,
                    metadata={
                        'interview_type':types,
                        'company':company,
                        'job':job,
                    },
                    metadata_seperator = "::",
                    metadata_template = "{key}=>{value}",
                    text_template = "Metadata: {metadata_str}\n-----\nContent: {content}",
                )
                full_doc.append(doc)
    pipeline = IngestionPipeline(
        transformations=[
            Settings.embed_model,
        ],
        vector_store=vector_store,
        docstore=SimpleDocumentStore()
    )
    pipeline.run(
        documents=full_doc,  # 确保传入列表
        in_place=True,
        show_progress=True,
    )
    print(collection.count())
def get_fitter(job:str)->dict:
    prompt_template="""
    角色：
    你是一个智能助手 ，帮助用户解决问题。
    任务：
    你需要把用户传入的工作岗位信息整合成json格式的信息用于检索数据库。
    工作流：
    1.首先仔细审视理解用户传入的岗位信息。
    2.提取出三方面内容 1.应聘公司名称 2.应聘岗位 
    3.返回json格式信息
    数据库中存在的应聘公司有：
    {company_name}
    数据库中存在的岗位有：
    {job_name}
    
    格式要求：
    你需要严格按照json的格式输出，不要添加额外字符，存在两个键 "company", "job".
    company 对应的类型为 str 表示从用户提供的岗位中提取到的应聘公司名称（从数据库中存在的名称中选择一个与用户输入语义最相关的公司，如果用户的输入与任何选项都不相关，则返回空值。）
    job 对应的类型为 str 表示从用户提供的岗位中提取到的职位名称（从数据库中存在的名称中选择一个与用户输入语义最相关的职位，如果用户的输入与任何选项都不相关，则返回空值。）
    用户输入：
    {job}
    """
    prompt=ChatPromptTemplate.from_template(prompt_template)
    chain=prompt|llm|JsonOutputParser()
    result=chain.invoke({'company_name':company_name,'job_name':job_name,'job':job})
    print(result)
    if isinstance(result,dict):
        return result
    return {'company':'','job':''}
def get_simility_interview_question(job:str,interview_type:str)->list[str]:
    print(111)
    index = VectorStoreIndex.from_vector_store(vector_store,embed_model=Settings.embed_model)
    fitter_name=get_fitter(job)
    # 创建一个精确匹配过滤器，key='company', value='阿里巴巴'
    # 这意味着我们只想要 metadata 中 'company' 字段是 '阿里巴巴' 的文档
    filters=[ExactMatchFilter(key="interview_type", value=interview_type)]
    if fitter_name['company']:
        company_filter=ExactMatchFilter(key="company", value=fitter_name['company'])
        print(company_filter)
        filters.append(company_filter)
    if fitter_name['job']:
        job_full_filter=ExactMatchFilter(key="job", value=fitter_name['job'])
        print(job_full_filter)
        filters.append(job_full_filter)
    filters=MetadataFilters(filters=filters)
    retriever = index.as_retriever(
        filters=filters,
        similarity_top_k=10  # 举例：获取最多10个结果
    )
    result=retriever.retrieve(job)
    print(len(result))
    print(result)
    rerank = SentenceTransformerRerank(
        model=rerank_model_name,
        top_n=4,
        device='cuda'  # 只保留重排序后的前3个结果
    )
    print(rerank.device)
    nodes = rerank.postprocess_nodes(nodes=result, query_str='1231')
    print(nodes)
