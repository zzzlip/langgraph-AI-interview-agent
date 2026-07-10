package com.aiinterview.platform.application.task.command;

import java.util.Objects;

/**
 * 取消任务命令。
 */
public record CancelTaskCommand(
        String userId,
        String taskId,
        String traceId
) {
    public CancelTaskCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
