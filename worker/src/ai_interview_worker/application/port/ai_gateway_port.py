"""AI Gateway Port — 外部模型/搜索/ASR/多模态能力的唯一出口。"""

from dataclasses import dataclass
from typing import Any, Protocol

from ai_interview_worker.domain.error import WorkerFailure


@dataclass(frozen=True)
class AiChatRequest:
    """文本模型调用请求。"""

    model: str = "deepseek-chat"
    messages: list[dict[str, str]] | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    provider: str = "deepseek"
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class AiEmbeddingRequest:
    """向量化请求。"""

    model: str = "text-embedding-v3"
    input: str = ""
    provider: str = "dashscope"


@dataclass(frozen=True)
class AiSearchRequest:
    """搜索请求。"""

    query: str = ""
    provider: str = "search"


@dataclass(frozen=True)
class AiTranscribeRequest:
    """ASR 语音转文字请求。"""

    audio_data: bytes = b""
    audio_format: str = "mp3"
    provider: str = "dashscope"


@dataclass(frozen=True)
class AiMultimodalRequest:
    """多模态分析请求。"""

    model: str = "qwen-vl-plus"
    image_data: bytes | None = None
    image_mime_type: str = "image/png"
    audio_data: bytes | None = None
    audio_format: str = "mp3"
    video_data: bytes | None = None
    video_format: str = "mp4"
    prompt: str = ""
    provider: str = "dashscope"


@dataclass(frozen=True)
class AiGatewayResult:
    """AI Gateway 调用结果。"""

    success: bool
    data: Any = None
    failure: WorkerFailure | None = None
    provider: str = ""


class AiGatewayPort(Protocol):
    """AI Gateway 统一入口 Port。"""

    def chat_completion(self, request: AiChatRequest) -> AiGatewayResult:
        """文本推理。"""
        ...

    def embedding(self, request: AiEmbeddingRequest) -> AiGatewayResult:
        """向量化。"""
        ...

    def search(self, request: AiSearchRequest) -> AiGatewayResult:
        """外部搜索。"""
        ...

    def transcribe_audio(self, request: AiTranscribeRequest) -> AiGatewayResult:
        """ASR 语音转文字。"""
        ...

    def analyze_multimodal(self, request: AiMultimodalRequest) -> AiGatewayResult:
        """多模态分析。"""
        ...
