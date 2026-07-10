package com.aiinterview.platform.application.asyncmessage.dto;

import java.time.Instant;
import java.util.Objects;

/**
 * 管理员异步链路摘要。
 */
public record AsyncTaskTraceResult(
        String taskId,
        String taskType,
        int currentAttempt,
        DeliverySummary delivery,
        ConsumptionSummary consumption,
        FeedbackSummary latestFeedback,
        DeadLetterSummary deadLetter
) {
    public AsyncTaskTraceResult {
        Objects.requireNonNull(taskId, "taskId 不能为空");
    }

    public record DeliverySummary(
            String status,
            int retryCount,
            String lastError,
            Instant createdAt,
            Instant deliveredAt
    ) {
    }

    public record ConsumptionSummary(
            String status,
            Instant receivedAt,
            Instant processedAt
    ) {
    }

    public record FeedbackSummary(
            String eventType,
            String status,
            String stage,
            Integer progress,
            Instant occurredAt
    ) {
    }

    public record DeadLetterSummary(
            String deadLetterId,
            String sourceType,
            String failureStage,
            String failureReason,
            boolean retryable,
            boolean manualHandled,
            Instant createdAt
    ) {
    }
}
