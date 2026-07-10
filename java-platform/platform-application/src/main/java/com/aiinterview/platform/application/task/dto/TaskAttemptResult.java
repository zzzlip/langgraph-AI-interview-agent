package com.aiinterview.platform.application.task.dto;

import java.time.Instant;

/**
 * 任务执行尝试结果。
 */
public record TaskAttemptResult(
        int attemptNo,
        String status,
        Instant startedAt,
        Instant finishedAt,
        Instant failedAt
) {
}
