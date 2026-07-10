package com.aiinterview.platform.application.asyncmessage.dto;

import java.time.Instant;

/**
 * 管理员死信列表条目。
 */
public record DeadLetterItemResult(
        String deadLetterId,
        String sourceType,
        String taskId,
        String taskType,
        int attempt,
        String failureStage,
        String failureReason,
        boolean retryable,
        boolean manualHandled,
        Instant createdAt
) {
}
