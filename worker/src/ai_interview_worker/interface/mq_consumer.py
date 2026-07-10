"""RabbitMQ 任务命令消费者。"""

import json
import signal
from collections.abc import Callable

import pika

from ai_interview_worker.application.worker_service import HandleDecisionType, HandleResult
from ai_interview_worker.infrastructure.logging import get_logger
from ai_interview_worker.infrastructure.rabbitmq.topology import (
    WORKER_COMMAND_QUEUE,
    declare_topology,
)

logger = get_logger(__name__)


class MessageConsumer:
    """RabbitMQ 任务命令消费者，负责消息接入与 ACK/NACK 协调。"""

    def __init__(
        self,
        *,
        connection_url: str,
        handle_message: Callable[[bytes], HandleResult] | None = None,
        handle_message_factory: Callable[
            [pika.adapters.blocking_connection.BlockingChannel], Callable[[bytes], HandleResult]
        ]
        | None = None,
        prefetch_count: int = 1,
    ) -> None:
        self.connection_url = connection_url
        self.handle_message = handle_message
        self.handle_message_factory = handle_message_factory
        self.prefetch_count = prefetch_count
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.adapters.blocking_connection.BlockingChannel | None = None
        self._running = False

    def start(self) -> None:
        """声明拓扑并开始消费。"""
        signal.signal(signal.SIGTERM, lambda _signo, _frame: self.stop())
        parameters = pika.URLParameters(self.connection_url)
        self._connection = pika.BlockingConnection(parameters)
        self._channel = self._connection.channel()
        if self.handle_message_factory is not None:
            self.handle_message = self.handle_message_factory(self._channel)
        handle_message = self.handle_message
        if handle_message is None:
            raise RuntimeError("handle_message is required")
        self._channel.basic_qos(prefetch_count=self.prefetch_count)
        declare_topology(self._channel)
        self._channel.basic_consume(
            queue=WORKER_COMMAND_QUEUE,
            on_message_callback=self._on_message,
            auto_ack=False,
        )
        self._running = True
        try:
            self._channel.start_consuming()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._channel and self._channel.is_open:
            self._channel.stop_consuming()
            self._channel.close()
        if self._connection and self._connection.is_open:
            self._connection.close()

    @property
    def is_running(self) -> bool:
        return self._running

    def _on_message(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            handle_message = self.handle_message
            if handle_message is None:
                raise RuntimeError("handle_message is required")
            result = handle_message(body)
            if result.decision_type is HandleDecisionType.ACK:
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return
            if result.decision_type is HandleDecisionType.NACK_REQUEUE:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                return
            if result.decision_type is HandleDecisionType.PROCESS and not result.entered_ai_execution:
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return

            logger.warning("message_left_without_terminal_feedback", reason=result.reason)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        except json.JSONDecodeError:
            logger.warning("invalid_json_received", body_preview=body[:200])
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:
            logger.error("message_processing_failed", error=str(exc)[:200])
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
