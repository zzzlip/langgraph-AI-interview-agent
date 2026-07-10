package com.aiinterview.platform.application.task.command;

import java.util.Objects;

/**
 * Worker 成功反馈命令。
 */
public record WorkerResultCommand(
        String taskId,
        int attempt,
        String resultPayload,
        String traceId
) {
    public WorkerResultCommand {
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(resultPayload, "resultPayload 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
