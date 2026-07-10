"""
Worker CLI 入口。

用法:
    python -m ai_interview_worker.interface.cli health
    python -m ai_interview_worker.interface.cli consume
"""
import sys
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from ai_interview_worker.application.ability_handler import AbilityRouter
from ai_interview_worker.application.algorithm_test.handler import AlgorithmTestHandler
from ai_interview_worker.application.mock_interview.handler import MockInterviewHandler
from ai_interview_worker.application.resume_evaluation.handler import ResumeEvaluationHandler
from ai_interview_worker.application.resume_optimization.handler import ResumeOptimizationHandler
from ai_interview_worker.application.worker_service import HandleResult, WorkerCommandService
from ai_interview_worker.domain.task_type import TaskType
from ai_interview_worker.infrastructure.logging import configure_logging, get_logger
from ai_interview_worker.infrastructure.rabbitmq.publisher import FeedbackPublisher
from ai_interview_worker.infrastructure.settings import Settings
from ai_interview_worker.infrastructure.sqlite.processed_command_store import SQLiteProcessedCommandStore
from ai_interview_worker.interface.mq_consumer import MessageConsumer


def main() -> None:
    """CLI 主入口"""
    configure_logging()
    logger = get_logger(__name__)

    if len(sys.argv) < 2:
        print("用法: python -m ai_interview_worker.interface.cli <health|consume>", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "health":
        _health_check(logger)
    elif command in {"run", "consume"}:
        _start_run(logger)
    else:
        print(f"未知命令: {command}", file=sys.stderr)
        sys.exit(1)


def _health_check(logger: Any) -> None:
    """执行健康自检 — 不连接真实外部服务"""
    logger.info("health_check_start")

    settings = Settings.from_env()
    missing = settings.validate()
    if missing:
        _print_config_errors(missing)
        sys.exit(1)

    # 检查基础导入
    try:
        from ai_interview_worker.domain.command import TaskCommand  # noqa: F401
        from ai_interview_worker.domain.error import TaskError  # noqa: F401
        from ai_interview_worker.domain.result import TaskResult  # noqa: F401
        from ai_interview_worker.interface.mq_consumer import MessageConsumer  # noqa: F401
        logger.info("domain_and_application_imports_ok")
    except ImportError as e:
        logger.error("import_failed", error=str(e))
        sys.exit(1)

    logger.info("health_check_complete", status="ok", missing_config_count=len(missing))


def _start_run(logger: Any) -> None:
    """执行启动前置校验并启动 RabbitMQ 消费循环。"""
    settings = Settings.from_env()
    missing = settings.validate()
    if missing:
        _print_config_errors(missing)
        sys.exit(1)

    consumer = MessageConsumer(
        connection_url=settings.rabbitmq_url,
        handle_message_factory=lambda channel: _build_message_handler(settings, channel),
        prefetch_count=1,
    )
    try:
        logger.info(
            "worker_startup_config_ok",
            enabled_providers=",".join(settings.enabled_providers),
            processing_timeout_seconds=settings.processing_timeout_seconds,
        )
        consumer.start()
    except Exception as exc:
        logger.error("worker_start_failed", error=str(exc)[:200])
        sys.exit(1)


def _print_config_errors(config_names: list[str]) -> None:
    """只输出配置项名称，避免泄露配置值。"""
    for config_name in config_names:
        print(f"config_invalid: {config_name}", file=sys.stderr)


def _build_message_handler(settings: Settings, channel: Any) -> Callable[[bytes], HandleResult]:
    """组装完整命令处理闭环，供 RabbitMQ 消费回调调用。"""
    idempotency_store = SQLiteProcessedCommandStore(settings.sqlite_path)
    feedback_publisher = FeedbackPublisher(channel)
    minio_adapter_cls = import_module("ai_interview_worker.infrastructure.minio.adapter").MinioFileAdapter
    ai_gateway_cls = import_module("ai_interview_worker.infrastructure.ai_gateway.gateway").HttpxAiGateway
    codeforces_module = import_module("ai_interview_worker.infrastructure.codeforces.client")
    artifact_adapter = minio_adapter_cls(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
    )
    ai_gateway = ai_gateway_cls(
        deepseek_api_key=settings.deepseek_api_key,
        dashscope_api_key=settings.dashscope_api_key,
        serper_api_key=settings.serper_api_key,
        enabled_providers=settings.enabled_providers,
        clock=lambda: datetime.now(timezone.utc),
    )
    codeforces_client = codeforces_module.HttpxCodeforcesClient(
        config=codeforces_module.CodeforcesClientConfig(
            timeout_seconds=settings.codeforces_timeout_seconds,
            max_submissions=settings.codeforces_max_submissions,
        )
    )
    router = AbilityRouter(
        handlers={
            TaskType.RESUME_EVALUATION: ResumeEvaluationHandler(),
            TaskType.RESUME_OPTIMIZATION: ResumeOptimizationHandler(),
            TaskType.ALGORITHM_TEST: AlgorithmTestHandler(
                codeforces_client=codeforces_client,
                max_submissions=settings.codeforces_max_submissions,
            ),
            TaskType.MOCK_INTERVIEW: MockInterviewHandler(),
        }
    )
    service = WorkerCommandService(
        idempotency_store=idempotency_store,
        feedback_publisher=feedback_publisher,
        worker_id=settings.worker_id,
        event_id_supplier=lambda: str(uuid.uuid4()),
        clock=lambda: datetime.now(timezone.utc),
        timeout_seconds=settings.processing_timeout_seconds,
        ability_router=router,
        minio_reader=artifact_adapter,
        artifact_writer=artifact_adapter,
        ai_gateway=ai_gateway,
    )
    return service.handle


if __name__ == "__main__":
    main()
