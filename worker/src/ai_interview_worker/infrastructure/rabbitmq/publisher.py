"""Worker 反馈事件 RabbitMQ 发布适配器。"""

import json
from typing import Any

import pika

from ai_interview_worker.application.feedback_service import FeedbackPublishError
from ai_interview_worker.infrastructure.rabbitmq.topology import FEEDBACK_EXCHANGE, get_feedback_routing_key


class FeedbackPublisher:
    """Worker 反馈事件发布器。"""

    def __init__(self, channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
        self.channel = channel
        channel.confirm_delivery()

    def publish(self, event: dict[str, Any]) -> None:
        """发布反馈事件到 feedback exchange，等待 Broker 确认。"""
        task_type = event.get("task_type", "unknown")
        routing_key = get_feedback_routing_key(task_type)
        body = json.dumps(event, ensure_ascii=False)
        try:
            confirmed = self.channel.basic_publish(
                exchange=FEEDBACK_EXCHANGE,
                routing_key=routing_key,
                body=body.encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    content_encoding="utf-8",
                    message_id=event.get("event_id", ""),
                    delivery_mode=2,
                ),
                mandatory=True,
            )
        except Exception as exc:
            raise FeedbackPublishError("feedback publish failed") from exc
        if not confirmed:
            raise FeedbackPublishError("feedback publish not confirmed by broker")
