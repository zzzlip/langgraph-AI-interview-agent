package com.aiinterview.platform.domain.asyncmessage;

import java.util.List;
import java.util.Optional;

/**
 * Inbox 事件仓储接口。
 */
public interface InboxEventRepository {

    void save(InboxEvent event);

    void update(InboxEvent event);

    Optional<InboxEvent> findById(String id);

    Optional<InboxEvent> findByEventId(String eventId);

    List<InboxEvent> findByTaskIdAndAttemptNo(String taskId, int attemptNo);

    Optional<InboxEvent> findConsumedByTaskIdAndAttemptNo(String taskId, int attemptNo);
}
