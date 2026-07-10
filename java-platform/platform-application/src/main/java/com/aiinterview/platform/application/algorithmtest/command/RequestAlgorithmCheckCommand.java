package com.aiinterview.platform.application.algorithmtest.command;

import java.util.Objects;

/**
 * 发起算法测试提交检查命令。
 */
public record RequestAlgorithmCheckCommand(
        String userId,
        String taskId,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public RequestAlgorithmCheckCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
