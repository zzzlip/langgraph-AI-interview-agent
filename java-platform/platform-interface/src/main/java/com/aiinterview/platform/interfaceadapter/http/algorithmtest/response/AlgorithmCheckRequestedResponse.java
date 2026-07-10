package com.aiinterview.platform.interfaceadapter.http.algorithmtest.response;

import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmCheckRequestedResult;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 算法测试提交检查响应。
 */
public record AlgorithmCheckRequestedResponse(
        @JsonProperty("task_id") String taskId,
        String status,
        @JsonProperty("status_label") String statusLabel,
        int attempt
) {
    public static AlgorithmCheckRequestedResponse from(AlgorithmCheckRequestedResult result) {
        return new AlgorithmCheckRequestedResponse(
                result.taskId(),
                result.status(),
                result.statusLabel(),
                result.attempt()
        );
    }
}
