package com.aiinterview.platform.domain.task;

import java.time.Instant;
import java.util.Objects;

/**
 * 任务失败值对象。
 */
public class TaskFailure {

    private final String failureStage;
    private final String failureCode;
    private final String userMessage;
    private final String adminSummary;
    private final boolean retryable;
    private final Instant failedAt;

    public TaskFailure(
            String failureStage,
            String failureCode,
            String userMessage,
            String adminSummary,
            boolean retryable,
            Instant failedAt
    ) {
        this.failureStage = validateRequiredText(failureStage, "failureStage", 64);
        this.failureCode = failureCode;
        this.userMessage = Objects.requireNonNull(userMessage, "userMessage 不能为空");
        this.adminSummary = Objects.requireNonNull(adminSummary, "adminSummary 不能为空");
        this.retryable = retryable;
        this.failedAt = Objects.requireNonNull(failedAt, "failedAt 不能为空");
    }

    private static String validateRequiredText(String value, String fieldName, int maxLength) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(fieldName + " 不能为空");
        }
        if (value.length() > maxLength) {
            throw new IllegalArgumentException(fieldName + " 最长 " + maxLength + " 字符");
        }
        return value;
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

    public boolean isRetryable() {
        return retryable;
    }

    public Instant getFailedAt() {
        return failedAt;
    }
}
