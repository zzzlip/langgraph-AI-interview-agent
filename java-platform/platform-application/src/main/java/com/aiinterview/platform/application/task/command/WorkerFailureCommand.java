package com.aiinterview.platform.application.task.command;

import java.util.Objects;

/**
 * Worker 失败反馈命令。
 */
public record WorkerFailureCommand(
        String taskId,
        int attempt,
        String failureStage,
        String failureCode,
        String userMessage,
        String adminSummary,
        boolean retryable,
        String traceId
) {
    public WorkerFailureCommand {
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(failureStage, "failureStage 不能为空");
        Objects.requireNonNull(userMessage, "userMessage 不能为空");
        Objects.requireNonNull(adminSummary, "adminSummary 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
