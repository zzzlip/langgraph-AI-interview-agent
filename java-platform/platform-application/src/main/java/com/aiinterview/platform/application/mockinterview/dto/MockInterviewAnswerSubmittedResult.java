package com.aiinterview.platform.application.mockinterview.dto;

/**
 * 模拟面试回答提交结果。
 */
public record MockInterviewAnswerSubmittedResult(
        String taskId,
        String status,
        String statusLabel,
        int attempt
) {
}
