"""
JSON 日志配置。

使用 structlog 输出结构化日志。
日志字段约定：
- trace_id: 请求追踪 ID
- task_id: 任务 ID
- attempt: 重试次数
- stage: 执行阶段
- event_id: 事件 ID
- worker_name: Worker 名称
- worker_version: Worker 版本
- error_code: 错误码
"""
from typing import cast

import structlog

from ai_interview_worker.infrastructure.settings import Settings


def configure_logging() -> None:
    """配置 structlog JSON 日志"""
    settings = Settings.from_env()
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.env == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
