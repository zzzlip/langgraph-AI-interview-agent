package com.aiinterview.platform.application.task.command;

import com.aiinterview.platform.domain.task.TaskStatus;
import java.util.Objects;

/**
 * Worker 进度反馈命令。
 */
public record WorkerProgressCommand(
        String taskId,
        int attempt,
        TaskStatus targetStatus,
        String stage,
        Integer progress,
        String message,
        String traceId
) {
    public WorkerProgressCommand {
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(targetStatus, "targetStatus 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
