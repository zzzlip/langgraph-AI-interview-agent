package com.aiinterview.platform.domain.asyncmessage;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * Outbox 事件仓储接口。
 */
public interface OutboxEventRepository {

    void save(OutboxEvent event);

    void update(OutboxEvent event);

    boolean updateByIdAndStatus(OutboxEvent event, AsyncDeliveryStatus expectedStatus);

    Optional<OutboxEvent> findById(String id);

    Optional<OutboxEvent> findByEventId(String eventId);

    Optional<OutboxEvent> findByTaskIdAndAttemptNoAndEventType(String taskId, int attemptNo, String eventType);

    List<OutboxEvent> findByTaskId(String taskId);

    List<OutboxEvent> findPublishable(Instant now, int limit);

    List<OutboxEvent> findStaleDelivering(Instant before, int limit);
}
