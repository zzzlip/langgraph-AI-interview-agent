package com.aiinterview.platform.application.task.dto;

/**
 * Worker 反馈处理结果。
 */
public record WorkerFeedbackResult(
        String taskId,
        String status,
        boolean ignored
) {
}
