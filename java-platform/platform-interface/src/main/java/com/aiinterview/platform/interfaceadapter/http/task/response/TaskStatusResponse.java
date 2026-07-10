package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskStatusResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 任务状态变更响应。
 */
public record TaskStatusResponse(
        @JsonProperty("task_id") String taskId,
        String status,
        @JsonProperty("cancelled_at") Instant cancelledAt,
        int attempt,
        @JsonProperty("retry_count") int retryCount,
        @JsonProperty("max_retry_count") int maxRetryCount
) {
    public static TaskStatusResponse from(TaskStatusResult result) {
        return new TaskStatusResponse(
                result.taskId(),
                result.status(),
                result.cancelledAt(),
                result.attempt(),
                result.retryCount(),
                result.maxRetryCount()
        );
    }
}
