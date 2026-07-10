package com.aiinterview.platform.interfaceadapter.http.mockinterview.response;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCreatedResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 模拟面试任务创建响应。
 */
public record MockInterviewCreatedResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("task_type") String taskType,
        String status,
        @JsonProperty("status_label") String statusLabel,
        @JsonProperty("created_at") Instant createdAt
) {

    public static MockInterviewCreatedResponse from(MockInterviewCreatedResult result) {
        return new MockInterviewCreatedResponse(
                result.taskId(),
                result.taskType(),
                result.status(),
                result.statusLabel(),
                result.createdAt()
        );
    }
}
