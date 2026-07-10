package com.aiinterview.platform.application.algorithmtest.command;

import java.util.Map;

/**
 * 算法测试题单中间反馈命令。
 */
public record AlgorithmProblemSetFeedbackCommand(
        String eventType,
        String taskId,
        String taskType,
        int attemptNo,
        String traceId,
        String status,
        String stage,
        Integer progress,
        String message,
        Map<String, Object> resultSummary
) {
}
