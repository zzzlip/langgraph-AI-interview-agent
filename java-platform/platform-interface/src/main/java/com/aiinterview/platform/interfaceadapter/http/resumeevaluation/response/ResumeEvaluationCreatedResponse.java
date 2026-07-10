package com.aiinterview.platform.interfaceadapter.http.resumeevaluation.response;

import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationCreatedResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 简历评估任务创建响应。
 */
public record ResumeEvaluationCreatedResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("task_type") String taskType,
        String status,
        String stage,
        Integer progress,
        @JsonProperty("created_at") Instant createdAt
) {
    public static ResumeEvaluationCreatedResponse from(ResumeEvaluationCreatedResult result) {
        return new ResumeEvaluationCreatedResponse(
                result.taskId(),
                result.taskType(),
                result.status(),
                null,
                null,
                result.createdAt()
        );
    }
}
