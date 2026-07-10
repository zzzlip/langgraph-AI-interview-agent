package com.aiinterview.platform.application.asyncmessage.dto;

import java.time.Instant;

/**
 * SSE 任务事件白名单数据。
 */
public record SseTaskEventResult(
        String eventName,
        String eventId,
        String taskId,
        String status,
        String stage,
        Integer progress,
        int attempt,
        String message,
        Boolean retryable,
        String traceId,
        Instant occurredAt
) {
}
