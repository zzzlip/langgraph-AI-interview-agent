package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.AdminFailedTaskItem;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 管理员失败任务列表项响应。
 */
public record AdminFailedTaskItemResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("user_id") String userId,
        @JsonProperty("task_type") String taskType,
        @JsonProperty("failure_stage") String failureStage,
        @JsonProperty("admin_summary") String adminSummary,
        int attempt,
        boolean retryable,
        @JsonProperty("last_failed_at") Instant lastFailedAt,
        @JsonProperty("trace_id") String traceId
) {
    static AdminFailedTaskItemResponse from(AdminFailedTaskItem item) {
        return new AdminFailedTaskItemResponse(
                item.taskId(),
                item.userId(),
                item.taskType(),
                item.failureStage(),
                item.adminSummary(),
                item.attempt(),
                item.retryable(),
                item.lastFailedAt(),
                item.traceId()
        );
    }
}
