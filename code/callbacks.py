import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler,AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class LLMBackHandler(AsyncCallbackHandler):
    def __init__(self,):
        self.start_time:float = 0
        self.end_time:float = 0
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.start_time = time.time()
        name=kwargs.get("name",'unknow')
        print(kwargs)
        print(f'当前处于的项目节点：{name}')
        print('\n--- LLM 调用开始---')
        print('llm输入：\n',prompts)
    def on_llm_end(self, response: LLMResult, **kwargs):
            print("\n--- LLM 调用结束 ---")
            self.end_time = time.time()
            print(f"生成的文本\n: {response.generations[0][0].text}\n")
            print(f'llm调用时长：{self.end_time - self.start_time}')

llm_callback_handler = [LLMBackHandler()]