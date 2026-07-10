package com.aiinterview.platform.application.mockinterview.dto;

import java.util.Map;
import java.util.Objects;

/**
 * 模拟面试当前等待状态。
 */
public record MockInterviewCurrentState(
        String taskId,
        int attemptNo,
        String currentQuestionId,
        String currentRoundId,
        String waitingInputType,
        long version,
        Map<String, Object> statePayload
) {
    public MockInterviewCurrentState {
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(currentQuestionId, "currentQuestionId 不能为空");
        Objects.requireNonNull(currentRoundId, "currentRoundId 不能为空");
        Objects.requireNonNull(waitingInputType, "waitingInputType 不能为空");
        Objects.requireNonNull(statePayload, "statePayload 不能为空");
    }
}
