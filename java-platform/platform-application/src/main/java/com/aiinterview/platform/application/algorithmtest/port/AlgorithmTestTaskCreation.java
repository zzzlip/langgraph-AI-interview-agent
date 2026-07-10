package com.aiinterview.platform.application.algorithmtest.port;

import java.util.Map;
import java.util.Objects;

/**
 * 算法测试任务创建参数。
 */
public record AlgorithmTestTaskCreation(
        String userId,
        String idempotencyKey,
        String requestHash,
        String traceId,
        String inputSummary,
        Map<String, Object> workerPayload
) {
    public AlgorithmTestTaskCreation {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
        Objects.requireNonNull(inputSummary, "inputSummary 不能为空");
        Objects.requireNonNull(workerPayload, "workerPayload 不能为空");
    }
}
