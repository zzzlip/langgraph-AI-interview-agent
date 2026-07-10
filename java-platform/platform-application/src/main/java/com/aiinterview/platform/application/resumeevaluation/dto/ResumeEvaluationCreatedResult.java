package com.aiinterview.platform.application.resumeevaluation.dto;

import java.time.Instant;

/**
 * 简历评估任务创建结果。
 */
public record ResumeEvaluationCreatedResult(
        String taskId,
        String taskType,
        String status,
        Instant createdAt
) {
}
