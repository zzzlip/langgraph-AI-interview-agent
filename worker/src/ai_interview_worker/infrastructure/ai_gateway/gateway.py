"""AI Gateway 统一入口实现，封装外部 provider 调用。"""

import base64
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx

from ai_interview_worker.application.port.ai_gateway_port import (
    AiChatRequest,
    AiEmbeddingRequest,
    AiGatewayPort,
    AiGatewayResult,
    AiMultimodalRequest,
    AiSearchRequest,
    AiTranscribeRequest,
)
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode

_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

_DEFAULT_TIMEOUTS: dict[str, int] = {
    "chat": 60,
    "embedding": 30,
    "search": 15,
    "transcribe": 120,
    "multimodal": 180,
}

_RETRY_COUNTS: dict[str, int] = {
    "chat": 2,
    "embedding": 1,
    "search": 1,
    "transcribe": 1,
    "multimodal": 1,
}

_DEFAULT_MODELS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "dashscope": "qwen-plus",
}

_LOG_SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie"}


class HttpxAiGateway(AiGatewayPort):
    """基于 httpx 的 AI Gateway 实现，封装 DeepSeek / DashScope 等 provider。"""

    def __init__(
        self,
        *,
        deepseek_api_key: str,
        dashscope_api_key: str,
        serper_api_key: str,
        enabled_providers: tuple[str, ...],
        clock: Callable[[], datetime],
        logger: Any = None,
    ) -> None:
        # api_key 允许为空，由任务级 llm_config 提供
        self.deepseek_api_key = deepseek_api_key.strip() if deepseek_api_key else ""
        self.dashscope_api_key = dashscope_api_key.strip() if dashscope_api_key else ""
        self.serper_api_key = serper_api_key.strip() if serper_api_key else ""
        self.enabled_providers = set(enabled_providers)
        self.clock = clock
        self._logger = logger

    def chat_completion(self, request: AiChatRequest) -> AiGatewayResult:
        if request.provider == "deepseek":
            if "deepseek" not in self.enabled_providers and not request.api_key:
                return self._provider_not_enabled(request.provider)
            return self._call_deepseek_chat(request)
        if request.provider == "dashscope":
            if "dashscope" not in self.enabled_providers:
                return self._provider_not_enabled(request.provider)
            return self._call_dashscope_chat(request)
        if request.api_key:
            return AiGatewayResult(
                success=False,
                failure=WorkerFailure(
                    WorkerFailureCode.INPUT_ERROR,
                    "任务级 LLM 配置当前仅支持 DeepSeek",
                    retryable=False,
                ),
                provider=request.provider,
            )
        if request.provider not in self.enabled_providers:
            return self._provider_not_enabled(request.provider)
        return AiGatewayResult(
            success=False,
            failure=WorkerFailure(
                WorkerFailureCode.INPUT_ERROR,
                "Chat provider 当前仅支持 deepseek 或 dashscope",
                retryable=False,
            ),
            provider=request.provider,
        )

    def embedding(self, request: AiEmbeddingRequest) -> AiGatewayResult:
        if request.provider not in self.enabled_providers:
            return self._provider_not_enabled(request.provider)
        return self._call_dashscope_embedding(request)

    def search(self, request: AiSearchRequest) -> AiGatewayResult:
        if request.provider not in self.enabled_providers:
            return self._provider_not_enabled(request.provider)
        return self._call_serper_search(request)

    def transcribe_audio(self, request: AiTranscribeRequest) -> AiGatewayResult:
        if request.provider not in self.enabled_providers:
            return self._provider_not_enabled(request.provider)
        return self._call_dashscope_asr(request)

    def analyze_multimodal(self, request: AiMultimodalRequest) -> AiGatewayResult:
        if request.provider not in self.enabled_providers:
            return self._provider_not_enabled(request.provider)
        return self._call_dashscope_multimodal(request)

    def _call_deepseek_chat(self, request: AiChatRequest) -> AiGatewayResult:
        url = _chat_completions_url(request.base_url or _DEEPSEEK_DEFAULT_BASE_URL)
        api_key = request.api_key or self.deepseek_api_key
        if not api_key.strip():
            return AiGatewayResult(
                success=False,
                failure=WorkerFailure(WorkerFailureCode.INPUT_ERROR, "DeepSeek API key 不能为空", retryable=False),
                provider=request.provider,
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": request.model or _DEFAULT_MODELS["deepseek"],
            "messages": request.messages or [],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        return self._do_request("chat", url, headers, payload, request.provider)

    def _call_dashscope_chat(self, request: AiChatRequest) -> AiGatewayResult:
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": request.model or _DEFAULT_MODELS["dashscope"],
            "messages": request.messages or [],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        return self._do_request("chat", url, headers, payload, request.provider)

    def _call_dashscope_embedding(self, request: AiEmbeddingRequest) -> AiGatewayResult:
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        headers = {
            "Authorization": f"Bearer {self.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": request.model, "input": {"text": request.input}}
        return self._do_request("embedding", url, headers, payload, request.provider)

    def _call_serper_search(self, request: AiSearchRequest) -> AiGatewayResult:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-Key": self.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": request.query}
        return self._do_request("search", url, headers, payload, request.provider)

    def _call_dashscope_asr(self, request: AiTranscribeRequest) -> AiGatewayResult:
        from dashscope.audio.asr import Recognition

        try:
            result = Recognition.call(
                api_key=self.dashscope_api_key,
                model="paraformer-realtime-v2",
                audio_data=request.audio_data,
                format=request.audio_format,
            )
            if result and result.get("output"):
                return AiGatewayResult(
                    success=True,
                    data={"text": result["output"].get("text", ""), "confidence": result["output"].get("confidence")},
                    provider=request.provider,
                )
            return AiGatewayResult(
                success=False,
                failure=WorkerFailure(WorkerFailureCode.EXTERNAL_SERVICE_ERROR, "语音识别返回空结果", retryable=True),
                provider=request.provider,
            )
        except Exception as ex:
            return self._to_failure(request.provider, str(ex))

    def _call_dashscope_multimodal(self, request: AiMultimodalRequest) -> AiGatewayResult:
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        content: list[dict[str, Any]] = []
        if request.prompt:
            content.append({"text": request.prompt})
        if request.image_data:
            content.append({"image": _data_url(request.image_data, request.image_mime_type)})
        if request.video_data:
            content.append({"video": _data_url(request.video_data, _video_mime_type(request.video_format))})
        if request.audio_data:
            content.append({"audio": _data_url(request.audio_data, _audio_mime_type(request.audio_format))})
        payload = {
            "model": request.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
        }
        return self._do_request("multimodal", url, headers, payload, request.provider)

    def _do_request(
        self,
        service_type: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        provider: str,
    ) -> AiGatewayResult:
        timeout = _DEFAULT_TIMEOUTS.get(service_type, 60)
        max_retries = _RETRY_COUNTS.get(service_type, 1)
        last_exception: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                # 每次重试新建短生命周期 client，避免把带鉴权头的请求对象留在可复用状态里。
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    if response.status_code == 429:
                        continue
                    if 500 <= response.status_code < 600 and attempt < max_retries:
                        continue
                    if response.status_code != 200:
                        return self._http_error(provider, response.status_code)
                    data = response.json()
                    if service_type == "chat":
                        data = _extract_chat_content(data)
                    return AiGatewayResult(
                        success=True,
                        data=data,
                        provider=provider,
                    )
            except httpx.TimeoutException as exc:
                last_exception = exc
                continue
            except httpx.ConnectError as exc:
                last_exception = exc
                continue
            except Exception:
                self._log_redacted("gateway_request_failed", service_type=service_type, provider=provider)
                return AiGatewayResult(
                    success=False,
                    failure=WorkerFailure(WorkerFailureCode.EXTERNAL_SERVICE_ERROR, "外部服务调用异常", retryable=True),
                    provider=provider,
                )

        error_msg = str(last_exception) if last_exception else "max retries exceeded"
        if "Timeout" in error_msg:
            return self._to_failure(provider, "外部服务超时")
        return self._to_failure(provider, "外部服务暂时不可用")

    def _http_error(self, provider: str, status_code: int) -> AiGatewayResult:
        if 400 <= status_code < 500 and status_code not in (429,):
            return AiGatewayResult(
                success=False,
                failure=WorkerFailure(
                    WorkerFailureCode.INPUT_ERROR, f"外部服务返回参数错误 ({status_code})", retryable=False
                ),
                provider=provider,
            )
        return self._to_failure(provider, f"外部服务返回 {status_code}")

    def _to_failure(self, provider: str, message: str) -> AiGatewayResult:
        return AiGatewayResult(
            success=False,
            failure=WorkerFailure(WorkerFailureCode.EXTERNAL_SERVICE_ERROR, message, retryable=True),
            provider=provider,
        )

    def _provider_not_enabled(self, provider: str) -> AiGatewayResult:
        return AiGatewayResult(
            success=False,
            failure=WorkerFailure(
                WorkerFailureCode.INTERNAL_ERROR,
                f"AI 能力 {provider} 未启用",
                retryable=False,
            ),
            provider=provider,
        )

    def _log_redacted(self, event: str, **kwargs: Any) -> None:
        if self._logger:
            safe_kwargs = {key: value for key, value in kwargs.items() if key.lower() not in _LOG_SENSITIVE_HEADERS}
            self._logger.warning(event, **safe_kwargs)


def _chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions"


def _data_url(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _audio_mime_type(audio_format: str) -> str:
    normalized = audio_format.strip().lower()
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
        "webm": "audio/webm",
    }.get(normalized, "application/octet-stream")


def _video_mime_type(video_format: str) -> str:
    normalized = video_format.strip().lower()
    return {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "mov": "video/quicktime",
    }.get(normalized, "application/octet-stream")


def _extract_chat_content(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choices[0].get("text"), str):
            return choices[0]["text"]
    return data
