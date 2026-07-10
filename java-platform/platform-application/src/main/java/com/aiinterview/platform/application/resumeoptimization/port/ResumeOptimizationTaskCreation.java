package com.aiinterview.platform.application.resumeoptimization.port;

import com.aiinterview.platform.application.resumeoptimization.dto.ResumeOptimizationInputFile;
import java.util.Map;

/**
 * 简历优化任务创建参数。
 */
public record ResumeOptimizationTaskCreation(
        String userId,
        String resumeFileId,
        String avatarFileId,
        String idempotencyKey,
        String requestHash,
        String traceId,
        String inputSummary,
        ResumeOptimizationInputFile resumeFile,
        ResumeOptimizationInputFile avatarFile,
        Map<String, Object> workerPayload
) {
}
