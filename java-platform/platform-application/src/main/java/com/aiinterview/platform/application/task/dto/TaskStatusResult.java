package com.aiinterview.platform.application.task.dto;

import java.time.Instant;

/**
 * 任务状态变更结果。
 */
public record TaskStatusResult(
        String taskId,
        String status,
        Instant cancelledAt,
        int attempt,
        int retryCount,
        int maxRetryCount
) {
}
