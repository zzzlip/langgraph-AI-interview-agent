package com.aiinterview.platform.application.task.dto;

import com.aiinterview.platform.application.result.dto.ResultDetailResult;
import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * 任务详情结果。
 */
public record TaskDetailResult(
        String taskId,
        String taskType,
        String status,
        String statusLabel,
        String stage,
        Integer progress,
        int attempt,
        int retryCount,
        int maxRetryCount,
        boolean retryable,
        boolean cancellable,
        String inputSummary,
        String resultReferenceId,
        ResultDetailResult result,
        TaskFailureResult failure,
        List<TaskAttemptResult> attempts,
        List<TaskEventResult> events,
        Instant createdAt,
        Instant updatedAt
) {
    public TaskDetailResult {
        attempts = List.copyOf(Objects.requireNonNull(attempts, "attempts 不能为空"));
        events = List.copyOf(Objects.requireNonNull(events, "events 不能为空"));
    }
}
