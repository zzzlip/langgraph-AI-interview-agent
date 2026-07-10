package com.aiinterview.platform.domain.asyncmessage;

import java.time.Instant;
import java.util.Objects;

/**
 * 异步协作死信记录。
 */
public class DeadLetterRecord {

    private final String id;
    private final DeadLetterSourceType sourceType;
    private final String sourceId;
    private final String taskId;
    private final String taskType;
    private final int attemptNo;
    private final String failureStage;
    private final String failureReason;
    private final boolean retryable;
    private final boolean manualHandled;
    private final String traceId;
    private final Instant createdAt;
    private final Instant updatedAt;

    public DeadLetterRecord(
            String id,
            DeadLetterSourceType sourceType,
            String sourceId,
            String taskId,
            String taskType,
            int attemptNo,
            String failureStage,
            String failureReason,
            boolean retryable,
            boolean manualHandled,
            String traceId,
            Instant createdAt,
            Instant updatedAt
    ) {
        if (attemptNo < 1) {
            throw new IllegalArgumentException("attemptNo 必须从 1 开始");
        }
        this.id = requireText(id, "id 不能为空");
        this.sourceType = Objects.requireNonNull(sourceType, "sourceType 不能为空");
        this.sourceId = requireText(sourceId, "sourceId 不能为空");
        this.taskId = requireText(taskId, "taskId 不能为空");
        this.taskType = taskType;
        this.attemptNo = attemptNo;
        this.failureStage = failureStage;
        this.failureReason = requireText(failureReason, "failureReason 不能为空");
        this.retryable = retryable;
        this.manualHandled = manualHandled;
        this.traceId = requireText(traceId, "traceId 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.updatedAt = Objects.requireNonNull(updatedAt, "updatedAt 不能为空");
    }

    public DeadLetterRecord markHandled(Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (manualHandled) {
            throw new IllegalStateException("死信记录已处理");
        }
        return new DeadLetterRecord(id, sourceType, sourceId, taskId, taskType, attemptNo, failureStage,
                failureReason, retryable, true, traceId, createdAt, now);
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

    public DeadLetterSourceType getSourceType() {
        return sourceType;
    }

    public String getSourceId() {
        return sourceId;
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

    public String getFailureStage() {
        return failureStage;
    }

    public String getFailureReason() {
        return failureReason;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public boolean isManualHandled() {
        return manualHandled;
    }

    public String getTraceId() {
        return traceId;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }
}
