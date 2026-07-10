package com.aiinterview.platform.application.algorithmtest.port;

import java.util.Objects;

/**
 * 算法测试提交检查请求参数。
 */
public record AlgorithmTestCheckRequest(
        String userId,
        String taskId,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public AlgorithmTestCheckRequest {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
