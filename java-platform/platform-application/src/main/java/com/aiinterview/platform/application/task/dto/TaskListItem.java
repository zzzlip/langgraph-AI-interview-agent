package com.aiinterview.platform.application.task.dto;

import java.time.Instant;

/**
 * 任务列表项。
 */
public record TaskListItem(
        String taskId,
        String taskType,
        String status,
        String statusLabel,
        String stage,
        Integer progress,
        boolean retryable,
        int retryCount,
        int maxRetryCount,
        boolean cancellable,
        boolean hasResult,
        Instant createdAt,
        Instant updatedAt,
        Instant completedAt,
        Instant cancelledAt
) {
}
