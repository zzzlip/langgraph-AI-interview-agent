package com.aiinterview.platform.application.task.command;

import java.util.Objects;

/**
 * 普通用户重试任务命令。
 */
public record RetryTaskCommand(
        String userId,
        String taskId,
        String reason,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public RetryTaskCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
