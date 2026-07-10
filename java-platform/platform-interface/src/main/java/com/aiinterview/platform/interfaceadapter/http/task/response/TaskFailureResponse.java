package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskFailureResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 普通用户失败信息响应。
 */
public record TaskFailureResponse(
        String stage,
        String message,
        @JsonProperty("failed_at") Instant failedAt
) {
    static TaskFailureResponse from(TaskFailureResult result) {
        if (result == null) {
            return null;
        }
        return new TaskFailureResponse(result.stage(), result.message(), result.failedAt());
    }
}
