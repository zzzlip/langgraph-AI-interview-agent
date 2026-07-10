package com.aiinterview.platform.application.mockinterview.dto;

import java.time.Instant;

/**
 * 模拟面试任务创建结果。
 */
public record MockInterviewCreatedResult(
        String taskId,
        String taskType,
        String status,
        String statusLabel,
        Instant createdAt
) {
}
