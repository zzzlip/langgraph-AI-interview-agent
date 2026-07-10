package com.aiinterview.platform.application.file.dto;

import java.time.Instant;

/**
 * 管理员结果状态摘要。
 */
public record AdminResultStatusItem(
        String resultId,
        String taskId,
        int attemptNo,
        String resultType,
        String resultStatus,
        long reportFileCount,
        long outputFileCount,
        Instant createdAt,
        Instant completedAt
) {
}
