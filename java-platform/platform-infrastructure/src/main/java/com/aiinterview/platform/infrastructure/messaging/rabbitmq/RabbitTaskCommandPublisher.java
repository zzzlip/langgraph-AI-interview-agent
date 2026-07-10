package com.aiinterview.platform.infrastructure.messaging.rabbitmq;

import com.aiinterview.platform.application.asyncmessage.port.RabbitMessagePublisher;
import com.aiinterview.platform.domain.asyncmessage.OutboxEvent;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import org.springframework.amqp.AmqpException;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Component;

/**
 * 任务命令 RabbitMQ 发布适配器。
 */
@Component
public class RabbitTaskCommandPublisher implements RabbitMessagePublisher {

    private static final long CONFIRM_TIMEOUT_MILLIS = 30_000L;

    private final RabbitTemplate rabbitTemplate;

    public RabbitTaskCommandPublisher(RabbitTemplate rabbitTemplate) {
        this.rabbitTemplate = Objects.requireNonNull(rabbitTemplate, "rabbitTemplate 不能为空");
    }

    @Override
    public void publish(OutboxEvent event) {
        Objects.requireNonNull(event, "event 不能为空");
        Message message = toMessage(event);
        try {
            rabbitTemplate.invoke(operations -> {
                operations.send(RabbitMqTopologyConfig.TASK_COMMAND_EXCHANGE, event.getRoutingKey(), message);
                operations.waitForConfirmsOrDie(CONFIRM_TIMEOUT_MILLIS);
                return null;
            });
        } catch (AmqpException ex) {
            throw new IllegalStateException("RabbitMQ 不可用或 Publisher Confirm 失败", ex);
        }
    }

    private static Message toMessage(OutboxEvent event) {
        MessageProperties properties = new MessageProperties();
        properties.setContentType(MessageProperties.CONTENT_TYPE_JSON);
        properties.setContentEncoding(StandardCharsets.UTF_8.name());
        properties.setMessageId(event.getEventId());
        properties.setCorrelationId(event.getTraceId());
        properties.setHeader("event_type", event.getEventType());
        properties.setHeader("task_id", event.getTaskId());
        properties.setHeader("task_type", event.getTaskType());
        properties.setHeader("attempt_no", event.getAttemptNo());
        return new Message(event.getPayload().getBytes(StandardCharsets.UTF_8), properties);
    }
}
