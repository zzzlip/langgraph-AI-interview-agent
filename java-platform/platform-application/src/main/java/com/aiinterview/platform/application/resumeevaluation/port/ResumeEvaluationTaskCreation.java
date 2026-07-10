package com.aiinterview.platform.application.resumeevaluation.port;

import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationInputFile;
import java.util.Map;

/**
 * 简历评估任务创建参数。
 */
public record ResumeEvaluationTaskCreation(
        String userId,
        String resumeFileId,
        String idempotencyKey,
        String requestHash,
        String traceId,
        String inputSummary,
        ResumeEvaluationInputFile inputFile,
        Map<String, Object> workerPayload
) {
}
