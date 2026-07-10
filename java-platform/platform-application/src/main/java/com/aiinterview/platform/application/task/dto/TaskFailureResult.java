package com.aiinterview.platform.application.task.dto;

import java.time.Instant;

/**
 * 普通用户可见失败信息。
 */
public record TaskFailureResult(
        String stage,
        String message,
        Instant failedAt
) {
}
