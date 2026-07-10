package com.aiinterview.platform.interfaceadapter.http.mockinterview.response;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerSubmittedResult;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 模拟面试回答提交响应。
 */
public record MockInterviewAnswerSubmittedResponse(
        @JsonProperty("task_id") String taskId,
        String status,
        @JsonProperty("status_label") String statusLabel,
        int attempt
) {

    public static MockInterviewAnswerSubmittedResponse from(MockInterviewAnswerSubmittedResult result) {
        return new MockInterviewAnswerSubmittedResponse(
                result.taskId(),
                result.status(),
                result.statusLabel(),
                result.attempt()
        );
    }
}
