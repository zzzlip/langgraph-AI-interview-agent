package com.aiinterview.platform.application.algorithmtest.dto;

import java.time.Instant;

/**
 * 算法测试任务创建结果。
 */
public record AlgorithmTestCreatedResult(
        String taskId,
        String taskType,
        String status,
        String statusLabel,
        Instant createdAt
) {
}
