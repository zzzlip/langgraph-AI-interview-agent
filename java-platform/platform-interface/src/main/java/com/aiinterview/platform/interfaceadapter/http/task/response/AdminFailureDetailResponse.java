package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskFailureDetailResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * 管理员失败详情响应。
 */
public record AdminFailureDetailResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("failure_stage") String failureStage,
        @JsonProperty("failure_code") String failureCode,
        @JsonProperty("admin_summary") String adminSummary,
        List<TaskAttemptResponse> attempts,
        List<TaskEventResponse> events,
        @JsonProperty("trace_id") String traceId
) {
    public static AdminFailureDetailResponse from(TaskFailureDetailResult result) {
        return new AdminFailureDetailResponse(
                result.taskId(),
                result.failureStage(),
                result.failureCode(),
                result.adminSummary(),
                result.attempts().stream().map(TaskAttemptResponse::from).toList(),
                result.events().stream().map(TaskEventResponse::from).toList(),
                result.traceId()
        );
    }
}
