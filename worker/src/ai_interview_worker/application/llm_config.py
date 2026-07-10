"""任务级 LLM 配置与模型响应解析。"""

import json
from dataclasses import dataclass
from typing import Any

from ai_interview_worker.domain.command import TaskCommand
from ai_interview_worker.domain.error import WorkerFailure, WorkerFailureCode

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
SUPPORTED_PROVIDER = "deepseek"


@dataclass(frozen=True)
class TaskLlmConfig:
    """用户为单个任务选择的 DeepSeek 配置。"""

    provider: str
    base_url: str
    model: str
    api_key: str


def resolve_task_llm_config(command: TaskCommand) -> TaskLlmConfig | None:
    """从私有 payload 或模拟面试状态中解析任务级 LLM 配置。"""
    raw = _find_config(command)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _llm_failure("llm_config 格式非法")
    provider = str(raw.get("provider", SUPPORTED_PROVIDER)).strip().lower()
    base_url = str(raw.get("base_url", DEFAULT_DEEPSEEK_BASE_URL)).strip()
    model = str(raw.get("model", DEFAULT_DEEPSEEK_MODEL)).strip()
    api_key = str(raw.get("api_key", "")).strip()
    if provider != SUPPORTED_PROVIDER:
        raise _llm_failure("当前仅支持 deepseek 平台")
    if not _valid_base_url(base_url):
        raise _llm_failure("DeepSeek API 地址非法")
    if not model:
        raise _llm_failure("DeepSeek 模型不能为空")
    if not api_key:
        raise _llm_failure("DeepSeek API key 不能为空")
    return TaskLlmConfig(provider=provider, base_url=base_url.rstrip("/"), model=model, api_key=api_key)


def public_llm_config(config: TaskLlmConfig | None) -> dict[str, str] | None:
    """返回可进入用户可见 payload 的非敏感 LLM 配置。"""
    if config is None:
        return None
    return {
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
    }


def private_llm_config(config: TaskLlmConfig | None) -> dict[str, str] | None:
    """返回仅用于 Worker 命令和 Java 中间状态的完整 LLM 配置。"""
    if config is None:
        return None
    return {
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "api_key": config.api_key,
    }


def llm_request_kwargs(command: TaskCommand) -> dict[str, Any]:
    """为 AiChatRequest 提供任务级覆盖参数。"""
    config = resolve_task_llm_config(command)
    if config is None:
        return {}
    return {
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "api_key": config.api_key,
    }


def chat_text(data: Any) -> str:
    """兼容 FakeGateway、OpenAI-compatible 响应和纯文本响应。"""
    if isinstance(data, str):
        return _strip_code_fence(data.strip())
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return _strip_code_fence(message["content"].strip())
                if isinstance(first.get("text"), str):
                    return _strip_code_fence(first["text"].strip())
        output = data.get("output")
        if isinstance(output, dict):
            text = output.get("text") or output.get("content")
            if isinstance(text, str):
                return _strip_code_fence(text.strip())
    return ""


def chat_json(data: Any) -> dict[str, Any]:
    """解析模型 JSON 输出，兼容测试替身直接返回 dict。"""
    if isinstance(data, dict) and "choices" not in data and "output" not in data:
        return data
    text = chat_text(data)
    if not text:
        raise _llm_failure("模型返回为空")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _llm_failure("模型返回 JSON 格式非法") from exc
    if not isinstance(parsed, dict):
        raise _llm_failure("模型返回 JSON 格式非法")
    return parsed


def _find_config(command: TaskCommand) -> Any:
    payload = command.input_data or {}
    candidates = [
        payload.get("llm_config"),
        _nested(payload, "business_payload", "llm_config"),
        _nested(payload, "input_summary", "llm_config"),
        _nested(command.input_summary or {}, "llm_config"),
        _nested(command.input_summary or {}, "interview_state", "llm_config"),
    ]
    return next((candidate for candidate in candidates if candidate is not None), None)


def _nested(source: dict[str, Any], first: str, second: str | None = None) -> Any:
    value = source.get(first)
    if second is None:
        return value
    return value.get(second) if isinstance(value, dict) else None


def _strip_code_fence(value: str) -> str:
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return value


def _valid_base_url(value: str) -> bool:
    return value.startswith("https://api.deepseek.com") or value.startswith("https://")


def _llm_failure(message: str) -> ValueError:
    return ValueError(
        WorkerFailure(
            failure_code=WorkerFailureCode.INPUT_ERROR,
            user_message=message,
            failed_stage="llm_config",
            retryable=False,
        )
    )
