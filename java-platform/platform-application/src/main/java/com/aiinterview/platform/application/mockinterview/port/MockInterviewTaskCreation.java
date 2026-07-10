package com.aiinterview.platform.application.mockinterview.port;

import java.util.Map;
import java.util.Objects;

/**
 * 模拟面试任务创建参数。
 */
public record MockInterviewTaskCreation(
        String userId,
        String idempotencyKey,
        String requestHash,
        String traceId,
        String resumeFileId,
        String inputSummary,
        Map<String, Object> workerPayload
) {
    public MockInterviewTaskCreation {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
        Objects.requireNonNull(resumeFileId, "resumeFileId 不能为空");
        Objects.requireNonNull(inputSummary, "inputSummary 不能为空");
        Objects.requireNonNull(workerPayload, "workerPayload 不能为空");
    }
}
