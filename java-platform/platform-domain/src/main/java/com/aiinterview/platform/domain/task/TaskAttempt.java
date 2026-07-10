package com.aiinterview.platform.domain.task;

import java.time.Instant;
import java.util.Objects;

/**
 * 任务单次执行尝试。
 */
public class TaskAttempt {

    private final String id;
    private final String taskId;
    private final int attemptNo;
    private final TaskStatus status;
    private final Instant startedAt;
    private final Instant finishedAt;
    private final Instant failedAt;
    private final String failureStage;
    private final String failureCode;
    private final String userMessage;
    private final String adminSummary;
    private final Instant createdAt;
    private final Instant updatedAt;

    public TaskAttempt(
            String id,
            String taskId,
            int attemptNo,
            TaskStatus status,
            Instant startedAt,
            Instant finishedAt,
            Instant failedAt,
            String failureStage,
            String failureCode,
            String userMessage,
            String adminSummary,
            Instant createdAt,
            Instant updatedAt
    ) {
        if (attemptNo < 1) {
            throw new IllegalArgumentException("attemptNo 必须从 1 开始");
        }
        this.id = Objects.requireNonNull(id, "id 不能为空");
        this.taskId = Objects.requireNonNull(taskId, "taskId 不能为空");
        this.attemptNo = attemptNo;
        this.status = Objects.requireNonNull(status, "status 不能为空");
        this.startedAt = startedAt;
        this.finishedAt = finishedAt;
        this.failedAt = failedAt;
        this.failureStage = failureStage;
        this.failureCode = failureCode;
        this.userMessage = userMessage;
        this.adminSummary = adminSummary;
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.updatedAt = Objects.requireNonNull(updatedAt, "updatedAt 不能为空");
    }

    public static TaskAttempt createInitial(String id, String taskId, Instant createdAt) {
        return new TaskAttempt(
                id,
                taskId,
                1,
                TaskStatus.PENDING,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                createdAt,
                createdAt
        );
    }

    public TaskAttempt markFailed(
            String failureStage,
            String failureCode,
            String userMessage,
            String adminSummary,
            Instant failedAt
    ) {
        return new TaskAttempt(
                id,
                taskId,
                attemptNo,
                TaskStatus.FAILED,
                startedAt,
                null,
                Objects.requireNonNull(failedAt, "failedAt 不能为空"),
                failureStage,
                failureCode,
                userMessage,
                adminSummary,
                createdAt,
                failedAt
        );
    }

    public String getId() {
        return id;
    }

    public String getTaskId() {
        return taskId;
    }

    public int getAttemptNo() {
        return attemptNo;
    }

    public TaskStatus getStatus() {
        return status;
    }

    public Instant getStartedAt() {
        return startedAt;
    }

    public Instant getFinishedAt() {
        return finishedAt;
    }

    public Instant getFailedAt() {
        return failedAt;
    }

    public String getFailureStage() {
        return failureStage;
    }

    public String getFailureCode() {
        return failureCode;
    }

    public String getUserMessage() {
        return userMessage;
    }

    public String getAdminSummary() {
        return adminSummary;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }
}
