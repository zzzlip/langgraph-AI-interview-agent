package com.aiinterview.platform.application.asyncmessage.command;

/**
 * Worker 反馈事件 envelope 命令。
 */
public record WorkerFeedbackEventCommand(
        int schemaVersion,
        String eventId,
        String eventType,
        String taskId,
        String taskType,
        int attemptNo,
        String traceId,
        String rawPayload,
        String payloadSummary
) {
}
