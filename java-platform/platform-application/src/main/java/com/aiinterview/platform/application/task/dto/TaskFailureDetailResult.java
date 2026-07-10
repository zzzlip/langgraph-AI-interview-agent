package com.aiinterview.platform.application.task.dto;

import java.util.List;
import java.util.Objects;

/**
 * 管理员失败详情结果。
 */
public record TaskFailureDetailResult(
        String taskId,
        String failureStage,
        String failureCode,
        String adminSummary,
        List<TaskAttemptResult> attempts,
        List<TaskEventResult> events,
        String traceId
) {
    public TaskFailureDetailResult {
        attempts = List.copyOf(Objects.requireNonNull(attempts, "attempts 不能为空"));
        events = List.copyOf(Objects.requireNonNull(events, "events 不能为空"));
    }
}
