package com.aiinterview.platform.application.task.dto;

import java.time.Instant;

/**
 * 任务事件结果。
 */
public record TaskEventResult(
        String eventType,
        String message,
        Instant occurredAt
) {
}
