package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskListItem;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 任务列表项响应。
 */
public record TaskListItemResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("task_type") String taskType,
        String status,
        @JsonProperty("status_label") String statusLabel,
        String stage,
        Integer progress,
        boolean retryable,
        @JsonProperty("retry_count") int retryCount,
        @JsonProperty("max_retry_count") int maxRetryCount,
        boolean cancellable,
        @JsonProperty("has_result") boolean hasResult,
        @JsonProperty("created_at") Instant createdAt,
        @JsonProperty("updated_at") Instant updatedAt,
        @JsonProperty("completed_at") Instant completedAt,
        @JsonProperty("cancelled_at") Instant cancelledAt
) {
    static TaskListItemResponse from(TaskListItem item) {
        return new TaskListItemResponse(
                item.taskId(),
                item.taskType(),
                item.status(),
                item.statusLabel(),
                item.stage(),
                item.progress(),
                item.retryable(),
                item.retryCount(),
                item.maxRetryCount(),
                item.cancellable(),
                item.hasResult(),
                item.createdAt(),
                item.updatedAt(),
                item.completedAt(),
                item.cancelledAt()
        );
    }
}
