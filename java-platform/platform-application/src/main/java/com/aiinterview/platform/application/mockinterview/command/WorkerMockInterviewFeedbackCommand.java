package com.aiinterview.platform.application.mockinterview.command;

import java.util.Map;

/**
 * Worker 模拟面试中间反馈命令。
 */
public record WorkerMockInterviewFeedbackCommand(
        String eventType,
        String taskId,
        String taskType,
        int attemptNo,
        String traceId,
        String status,
        String stage,
        Integer progress,
        String message,
        Map<String, Object> payload
) {
}
