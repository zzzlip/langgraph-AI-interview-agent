package com.aiinterview.platform.application.resumeoptimization.dto;

import java.time.Instant;

/**
 * 简历优化任务创建结果。
 */
public record ResumeOptimizationCreatedResult(
        String taskId,
        String taskType,
        String status,
        Instant createdAt
) {
}
