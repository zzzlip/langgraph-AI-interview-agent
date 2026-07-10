package com.aiinterview.platform.interfaceadapter.http.resumeoptimization.response;

import com.aiinterview.platform.application.resumeoptimization.dto.ResumeOptimizationCreatedResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 简历优化任务创建响应。
 */
public record ResumeOptimizationCreatedResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("task_type") String taskType,
        String status,
        String stage,
        Integer progress,
        @JsonProperty("created_at") Instant createdAt
) {
    public static ResumeOptimizationCreatedResponse from(ResumeOptimizationCreatedResult result) {
        return new ResumeOptimizationCreatedResponse(
                result.taskId(),
                result.taskType(),
                result.status(),
                null,
                null,
                result.createdAt()
        );
    }
}
