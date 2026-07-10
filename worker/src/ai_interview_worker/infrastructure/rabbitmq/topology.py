"""RabbitMQ 拓扑声明 — 与 spec05 保持一致。"""

import pika

COMMAND_EXCHANGE = "ai.task.command"
FEEDBACK_EXCHANGE = "ai.task.feedback"
DLX_EXCHANGE = "ai.task.dlx"

WORKER_COMMAND_QUEUE = "ai.worker.task.command"
PLATFORM_FEEDBACK_QUEUE = "ai.platform.task.feedback"
PLATFORM_DEAD_LETTER_QUEUE = "ai.platform.task.dead-letter"

TASK_TYPES = ("resume_evaluation", "resume_optimization", "algorithm_test", "mock_interview")


def declare_topology(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    """声明 RabbitMQ 交换机和队列。"""
    channel.exchange_declare(exchange=COMMAND_EXCHANGE, exchange_type="direct", durable=True)
    channel.exchange_declare(exchange=FEEDBACK_EXCHANGE, exchange_type="direct", durable=True)
    channel.exchange_declare(exchange=DLX_EXCHANGE, exchange_type="direct", durable=True)

    channel.queue_declare(queue=WORKER_COMMAND_QUEUE, durable=True)
    channel.queue_declare(queue=PLATFORM_FEEDBACK_QUEUE, durable=True)
    channel.queue_declare(queue=PLATFORM_DEAD_LETTER_QUEUE, durable=True)

    for task_type in TASK_TYPES:
        channel.queue_bind(
            queue=WORKER_COMMAND_QUEUE,
            exchange=COMMAND_EXCHANGE,
            routing_key=f"task.{task_type}.command",
        )


def get_feedback_routing_key(task_type: str) -> str:
    """计算反馈消息的路由键。"""
    return f"task.{task_type}.feedback"
