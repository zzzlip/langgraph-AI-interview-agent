package com.aiinterview.platform.domain.asyncmessage;

import java.time.Instant;
import java.util.Objects;

/**
 * Worker 反馈 Inbox 事件。
 */
public class InboxEvent {

    private final String id;
    private final String eventId;
    private final String eventType;
    private final String taskId;
    private final String taskType;
    private final int attemptNo;
    private final WorkerFeedbackType feedbackType;
    private final AsyncConsumptionStatus status;
    private final String payloadHash;
    private final String payloadSummary;
    private final int retryCount;
    private final int maxRetryCount;
    private final String lastError;
    private final String traceId;
    private final Instant receivedAt;
    private final Instant processedAt;
    private final Instant createdAt;
    private final Instant updatedAt;

    public InboxEvent(
            String id,
            String eventId,
            String eventType,
            String taskId,
            String taskType,
            int attemptNo,
            WorkerFeedbackType feedbackType,
            AsyncConsumptionStatus status,
            String payloadHash,
            String payloadSummary,
            int retryCount,
            int maxRetryCount,
            String lastError,
            String traceId,
            Instant receivedAt,
            Instant processedAt,
            Instant createdAt,
            Instant updatedAt
    ) {
        if (attemptNo < 1) {
            throw new IllegalArgumentException("attemptNo 必须从 1 开始");
        }
        if (retryCount < 0) {
            throw new IllegalArgumentException("retryCount 不能小于 0");
        }
        if (maxRetryCount < 1) {
            throw new IllegalArgumentException("maxRetryCount 必须大于 0");
        }
        this.id = requireText(id, "id 不能为空");
        this.eventId = requireText(eventId, "eventId 不能为空");
        this.eventType = requireText(eventType, "eventType 不能为空");
        this.taskId = requireText(taskId, "taskId 不能为空");
        this.taskType = requireText(taskType, "taskType 不能为空");
        this.attemptNo = attemptNo;
        this.feedbackType = Objects.requireNonNull(feedbackType, "feedbackType 不能为空");
        this.status = Objects.requireNonNull(status, "status 不能为空");
        this.payloadHash = requireText(payloadHash, "payloadHash 不能为空");
        this.payloadSummary = payloadSummary;
        this.retryCount = retryCount;
        this.maxRetryCount = maxRetryCount;
        this.lastError = lastError;
        this.traceId = requireText(traceId, "traceId 不能为空");
        this.receivedAt = Objects.requireNonNull(receivedAt, "receivedAt 不能为空");
        this.processedAt = processedAt;
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.updatedAt = Objects.requireNonNull(updatedAt, "updatedAt 不能为空");
    }

    public boolean hasSamePayloadHash(String candidatePayloadHash) {
        return payloadHash.equals(candidatePayloadHash);
    }

    public InboxEvent markProcessing(Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status != AsyncConsumptionStatus.RECEIVED && status != AsyncConsumptionStatus.FAILED) {
            throw new IllegalStateException("当前 Inbox 状态不允许处理");
        }
        if (retryCount >= maxRetryCount) {
            throw new IllegalStateException("Inbox 消费次数已达上限");
        }
        return copy(AsyncConsumptionStatus.PROCESSING, retryCount + 1, lastError, processedAt, now);
    }

    public InboxEvent markProcessed(Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status != AsyncConsumptionStatus.PROCESSING) {
            throw new IllegalStateException("当前 Inbox 状态不允许标记已处理");
        }
        return copy(AsyncConsumptionStatus.PROCESSED, retryCount, null, now, now);
    }

    public InboxEvent markIgnored(String reason, Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status != AsyncConsumptionStatus.PROCESSING) {
            throw new IllegalStateException("当前 Inbox 状态不允许忽略");
        }
        return copy(AsyncConsumptionStatus.IGNORED, retryCount, reason, now, now);
    }

    public InboxEvent markFailed(String error, Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status != AsyncConsumptionStatus.PROCESSING) {
            throw new IllegalStateException("当前 Inbox 状态不允许标记失败");
        }
        return copy(AsyncConsumptionStatus.FAILED, retryCount, error, processedAt, now);
    }

    public InboxEvent markDead(String error, Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status.isFinal()) {
            throw new IllegalStateException("终态 Inbox 不能进入死信");
        }
        return copy(AsyncConsumptionStatus.DEAD, retryCount, error, processedAt, now);
    }

    private InboxEvent copy(
            AsyncConsumptionStatus nextStatus,
            int nextRetryCount,
            String nextLastError,
            Instant nextProcessedAt,
            Instant nextUpdatedAt
    ) {
        return new InboxEvent(id, eventId, eventType, taskId, taskType, attemptNo, feedbackType, nextStatus,
                payloadHash, payloadSummary, nextRetryCount, maxRetryCount, nextLastError, traceId, receivedAt,
                nextProcessedAt, createdAt, nextUpdatedAt);
    }

    private static String requireText(String value, String message) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    public String getId() {
        return id;
    }

    public String getEventId() {
        return eventId;
    }

    public String getEventType() {
        return eventType;
    }

    public String getTaskId() {
        return taskId;
    }

    public String getTaskType() {
        return taskType;
    }

    public int getAttemptNo() {
        return attemptNo;
    }

    public WorkerFeedbackType getFeedbackType() {
        return feedbackType;
    }

    public AsyncConsumptionStatus getStatus() {
        return status;
    }

    public String getPayloadHash() {
        return payloadHash;
    }

    public String getPayloadSummary() {
        return payloadSummary;
    }

    public int getRetryCount() {
        return retryCount;
    }

    public int getMaxRetryCount() {
        return maxRetryCount;
    }

    public String getLastError() {
        return lastError;
    }

    public String getTraceId() {
        return traceId;
    }

    public Instant getReceivedAt() {
        return receivedAt;
    }

    public Instant getProcessedAt() {
        return processedAt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }
}
