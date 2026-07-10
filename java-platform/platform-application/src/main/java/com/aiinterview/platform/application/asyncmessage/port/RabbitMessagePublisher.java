package com.aiinterview.platform.application.asyncmessage.port;

import com.aiinterview.platform.domain.asyncmessage.OutboxEvent;

/**
 * RabbitMQ 消息发布端口。
 */
public interface RabbitMessagePublisher {

    void publish(OutboxEvent event);
}
