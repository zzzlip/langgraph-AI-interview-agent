package com.aiinterview.platform.application.task.command;

import com.aiinterview.platform.domain.task.TaskType;
import java.util.Objects;

/**
 * 创建任务命令。
 */
public record CreateTaskCommand(
        String userId,
        TaskType taskType,
        String inputSummary,
        String idempotencyKey,
        String requestHash,
        String resourceType,
        String resourceId,
        String traceId
) {
    public CreateTaskCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(taskType, "taskType 不能为空");
        Objects.requireNonNull(inputSummary, "inputSummary 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(resourceType, "resourceType 不能为空");
        Objects.requireNonNull(resourceId, "resourceId 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
