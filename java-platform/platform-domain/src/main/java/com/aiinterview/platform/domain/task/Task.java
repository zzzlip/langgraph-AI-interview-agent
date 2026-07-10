package com.aiinterview.platform.domain.task;

import java.time.Instant;
import java.util.Objects;

/**
 * 任务中心聚合根，统一约束任务主状态流转。
 */
public class Task {

    private final String id;
    private final String userId;
    private final TaskType taskType;
    private TaskStatus status;
    private String stage;
    private Integer progress;
    private final String inputSummary;
    private String resultReferenceId;
    private int currentAttempt;
    private int retryCount;
    private final int maxRetryCount;
    private boolean retryable;
    private final long version;
    private final String traceId;
    private final Instant createdAt;
    private Instant updatedAt;
    private Instant completedAt;
    private Instant cancelledAt;
    private Instant lastFailedAt;

    public Task(
            String id,
            String userId,
            TaskType taskType,
            TaskStatus status,
            String stage,
            Integer progress,
            String inputSummary,
            String resultReferenceId,
            int currentAttempt,
            int retryCount,
            int maxRetryCount,
            boolean retryable,
            long version,
            String traceId,
            Instant createdAt,
            Instant updatedAt,
            Instant completedAt,
            Instant cancelledAt,
            Instant lastFailedAt
    ) {
        if (currentAttempt < 1) {
            throw new IllegalArgumentException("currentAttempt 必须从 1 开始");
        }
        if (retryCount < 0) {
            throw new IllegalArgumentException("retryCount 不能小于 0");
        }
        if (maxRetryCount < 0) {
            throw new IllegalArgumentException("maxRetryCount 不能小于 0");
        }
        validateProgress(progress);
        this.id = Objects.requireNonNull(id, "id 不能为空");
        this.userId = Objects.requireNonNull(userId, "userId 不能为空");
        this.taskType = Objects.requireNonNull(taskType, "taskType 不能为空");
        this.status = Objects.requireNonNull(status, "status 不能为空");
        this.stage = stage;
        this.progress = progress;
        this.inputSummary = Objects.requireNonNull(inputSummary, "inputSummary 不能为空");
        this.resultReferenceId = resultReferenceId;
        this.currentAttempt = currentAttempt;
        this.retryCount = retryCount;
        this.maxRetryCount = maxRetryCount;
        this.retryable = retryable;
        this.version = version;
        this.traceId = Objects.requireNonNull(traceId, "traceId 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.updatedAt = Objects.requireNonNull(updatedAt, "updatedAt 不能为空");
        this.completedAt = completedAt;
        this.cancelledAt = cancelledAt;
        this.lastFailedAt = lastFailedAt;
    }

    public static Task create(String id, String userId, TaskType taskType, String inputSummary, String traceId, Instant createdAt) {
        return new Task(
                id,
                userId,
                taskType,
                TaskStatus.PENDING,
                null,
                null,
                inputSummary,
                null,
                1,
                0,
                3,
                false,
                0L,
                traceId,
                createdAt,
                createdAt,
                null,
                null,
                null
        );
    }

    public void markQueued(Instant occurredAt) {
        requireTransition(TaskStatus.QUEUED, TaskStatus.PENDING, TaskStatus.WAITING_USER_INPUT);
        this.status = TaskStatus.QUEUED;
        this.retryable = false;
        this.updatedAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
    }

    public void startUserInputFollowupAttempt(Instant occurredAt) {
        requireTransition(TaskStatus.QUEUED, TaskStatus.WAITING_USER_INPUT);
        this.status = TaskStatus.QUEUED;
        this.currentAttempt += 1;
        this.updatedAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
    }

    public void markRunning(String stage, Integer progress, Instant occurredAt) {
        requireTransition(TaskStatus.RUNNING, TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.WAITING_USER_INPUT);
        validateProgress(progress);
        this.status = TaskStatus.RUNNING;
        this.stage = stage;
        this.progress = progress;
        this.retryable = false;
        this.updatedAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
    }

    public void markWaitingUserInput(String stage, String message, Instant occurredAt) {
        requireTransition(TaskStatus.WAITING_USER_INPUT, TaskStatus.RUNNING);
        if (message == null || message.isBlank()) {
            throw new IllegalArgumentException("等待用户补充必须包含说明");
        }
        this.status = TaskStatus.WAITING_USER_INPUT;
        this.stage = stage;
        this.retryable = false;
        this.updatedAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
    }

    public void markSuccess(String resultReferenceId, Instant occurredAt) {
        requireTransition(TaskStatus.SUCCESS, TaskStatus.RUNNING);
        this.status = TaskStatus.SUCCESS;
        this.progress = 100;
        this.resultReferenceId = Objects.requireNonNull(resultReferenceId, "resultReferenceId 不能为空");
        this.completedAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
        this.updatedAt = occurredAt;
        this.retryable = false;
    }

    public void markFailed(TaskFailure failure, Instant occurredAt) {
        requireTransition(
                TaskStatus.FAILED,
                TaskStatus.PENDING,
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.WAITING_USER_INPUT
        );
        Objects.requireNonNull(failure, "failure 不能为空");
        this.status = TaskStatus.FAILED;
        this.stage = failure.getFailureStage();
        this.retryable = failure.isRetryable();
        this.lastFailedAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
        this.updatedAt = occurredAt;
    }

    public void cancel(String operatorUserId, Instant occurredAt) {
        Objects.requireNonNull(operatorUserId, "operatorUserId 不能为空");
        requireTransition(TaskStatus.CANCELLED, TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.WAITING_USER_INPUT);
        this.status = TaskStatus.CANCELLED;
        this.cancelledAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
        this.updatedAt = occurredAt;
        this.retryable = false;
    }

    public void retry(String operator, Instant occurredAt) {
        Objects.requireNonNull(operator, "operator 不能为空");
        if (status != TaskStatus.FAILED) {
            throw new IllegalStateException("仅失败任务允许重试");
        }
        if (retryCount >= maxRetryCount) {
            throw new IllegalStateException("重试次数已达上限");
        }
        this.status = TaskStatus.QUEUED;
        this.currentAttempt += 1;
        this.retryCount += 1;
        this.retryable = false;
        this.updatedAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
    }

    public boolean isCancellable() {
        return status == TaskStatus.PENDING
                || status == TaskStatus.QUEUED
                || status == TaskStatus.RUNNING
                || status == TaskStatus.WAITING_USER_INPUT;
    }

    private void requireTransition(TaskStatus target, TaskStatus... allowedSources) {
        for (TaskStatus allowedSource : allowedSources) {
            if (status == allowedSource) {
                return;
            }
        }
        throw new IllegalStateException("任务状态 " + status.getCode() + " 不允许流转到 " + target.getCode());
    }

    private static void validateProgress(Integer progress) {
        if (progress != null && (progress < 0 || progress > 100)) {
            throw new IllegalArgumentException("progress 必须在 0 到 100 之间");
        }
    }

    public String getId() {
        return id;
    }

    public String getUserId() {
        return userId;
    }

    public TaskType getTaskType() {
        return taskType;
    }

    public TaskStatus getStatus() {
        return status;
    }

    public String getStage() {
        return stage;
    }

    public Integer getProgress() {
        return progress;
    }

    public String getInputSummary() {
        return inputSummary;
    }

    public String getResultReferenceId() {
        return resultReferenceId;
    }

    public int getCurrentAttempt() {
        return currentAttempt;
    }

    public int getRetryCount() {
        return retryCount;
    }

    public int getMaxRetryCount() {
        return maxRetryCount;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public long getVersion() {
        return version;
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

    public Instant getCompletedAt() {
        return completedAt;
    }

    public Instant getCancelledAt() {
        return cancelledAt;
    }

    public Instant getLastFailedAt() {
        return lastFailedAt;
    }
}
