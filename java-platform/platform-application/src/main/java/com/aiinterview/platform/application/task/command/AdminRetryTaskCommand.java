package com.aiinterview.platform.application.task.command;

import java.util.Objects;

/**
 * 管理员重试任务命令。
 */
public record AdminRetryTaskCommand(
        String adminId,
        String role,
        String taskId,
        String reason,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public AdminRetryTaskCommand {
        Objects.requireNonNull(adminId, "adminId 不能为空");
        Objects.requireNonNull(role, "role 不能为空");
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
