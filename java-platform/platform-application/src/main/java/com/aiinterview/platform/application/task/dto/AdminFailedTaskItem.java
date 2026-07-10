package com.aiinterview.platform.application.task.dto;

import java.time.Instant;

/**
 * 管理员失败任务列表项。
 */
public record AdminFailedTaskItem(
        String taskId,
        String userId,
        String taskType,
        String failureStage,
        String adminSummary,
        int attempt,
        boolean retryable,
        Instant lastFailedAt,
        String traceId
) {
}
