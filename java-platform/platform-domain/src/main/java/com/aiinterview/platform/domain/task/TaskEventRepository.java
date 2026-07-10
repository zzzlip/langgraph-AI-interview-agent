package com.aiinterview.platform.domain.task;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * 任务事件仓储接口。
 */
public interface TaskEventRepository {

    void save(TaskEvent event);

    Optional<TaskEvent> findById(String eventId);

    List<TaskEvent> findByTaskIdOrderByOccurredAt(String taskId);

    List<TaskEvent> findByTaskIdAndOccurredAtBetween(String taskId, Instant startInclusive, Instant endExclusive);

    List<TaskEvent> findByTraceId(String traceId);
}
