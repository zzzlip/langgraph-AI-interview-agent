package com.aiinterview.platform.interfaceadapter.http.asyncmessage.response;

import com.aiinterview.platform.application.asyncmessage.dto.AsyncTaskTraceResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 管理员异步链路响应。
 */
public record AsyncTaskTraceResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("task_type") String taskType,
        @JsonProperty("current_attempt") int currentAttempt,
        DeliveryResponse delivery,
        ConsumptionResponse consumption,
        @JsonProperty("latest_feedback") FeedbackResponse latestFeedback,
        @JsonProperty("dead_letter") DeadLetterResponse deadLetter
) {
    public static AsyncTaskTraceResponse from(AsyncTaskTraceResult result) {
        return new AsyncTaskTraceResponse(
                result.taskId(),
                result.taskType(),
                result.currentAttempt(),
                DeliveryResponse.from(result.delivery()),
                ConsumptionResponse.from(result.consumption()),
                FeedbackResponse.from(result.latestFeedback()),
                DeadLetterResponse.from(result.deadLetter())
        );
    }

    public record DeliveryResponse(String status, @JsonProperty("retry_count") int retryCount,
                                   @JsonProperty("last_error") String lastError,
                                   @JsonProperty("created_at") Instant createdAt,
                                   @JsonProperty("delivered_at") Instant deliveredAt) {
        static DeliveryResponse from(AsyncTaskTraceResult.DeliverySummary summary) {
            return summary == null ? null : new DeliveryResponse(summary.status(), summary.retryCount(),
                    summary.lastError(), summary.createdAt(), summary.deliveredAt());
        }
    }

    public record ConsumptionResponse(String status, @JsonProperty("received_at") Instant receivedAt,
                                      @JsonProperty("processed_at") Instant processedAt) {
        static ConsumptionResponse from(AsyncTaskTraceResult.ConsumptionSummary summary) {
            return summary == null ? null : new ConsumptionResponse(summary.status(), summary.receivedAt(),
                    summary.processedAt());
        }
    }

    public record FeedbackResponse(@JsonProperty("event_type") String eventType, String status, String stage,
                                   Integer progress, @JsonProperty("occurred_at") Instant occurredAt) {
        static FeedbackResponse from(AsyncTaskTraceResult.FeedbackSummary summary) {
            return summary == null ? null : new FeedbackResponse(summary.eventType(), summary.status(),
                    summary.stage(), summary.progress(), summary.occurredAt());
        }
    }

    public record DeadLetterResponse(@JsonProperty("dead_letter_id") String deadLetterId,
                                     @JsonProperty("source_type") String sourceType,
                                     @JsonProperty("failure_stage") String failureStage,
                                     @JsonProperty("failure_reason") String failureReason,
                                     boolean retryable,
                                     @JsonProperty("manual_handled") boolean manualHandled,
                                     @JsonProperty("created_at") Instant createdAt) {
        static DeadLetterResponse from(AsyncTaskTraceResult.DeadLetterSummary summary) {
            return summary == null ? null : new DeadLetterResponse(summary.deadLetterId(), summary.sourceType(),
                    summary.failureStage(), summary.failureReason(), summary.retryable(), summary.manualHandled(),
                    summary.createdAt());
        }
    }
}
