package com.aiinterview.platform.application.task.dto;

/**
 * 创建任务结果。
 */
public record TaskCreatedResult(
        String taskId,
        String status
) {
}
