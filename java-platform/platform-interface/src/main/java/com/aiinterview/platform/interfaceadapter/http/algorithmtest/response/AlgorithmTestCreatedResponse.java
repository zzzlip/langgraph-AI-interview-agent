package com.aiinterview.platform.interfaceadapter.http.algorithmtest.response;

import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmTestCreatedResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 算法测试任务创建响应。
 */
public record AlgorithmTestCreatedResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("task_type") String taskType,
        String status,
        @JsonProperty("status_label") String statusLabel,
        @JsonProperty("created_at") Instant createdAt
) {
    public static AlgorithmTestCreatedResponse from(AlgorithmTestCreatedResult result) {
        return new AlgorithmTestCreatedResponse(
                result.taskId(),
                result.taskType(),
                result.status(),
                result.statusLabel(),
                result.createdAt()
        );
    }
}
