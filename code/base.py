from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from openai import OpenAI,AsyncOpenAI
import api_key



llm_reasoner = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key.dp_api,
    model="deepseek-reasoner",
    temperature=0.7,
    streaming=True,
)
llm=ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key.dp_api,
    model="deepseek-chat",
    temperature=0.7,
    streaming=True,
)
llm_qwen=ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key.qwen_api,
    model="qwen-max-latest",
    temperature=0.7,
    streaming=True,
)
video_client= AsyncOpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key.qwen_api,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
llm_google=llm_qwen
# llm_reasoner=llm_qwen
# llm=llm_qwen