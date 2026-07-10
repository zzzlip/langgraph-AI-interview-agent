package com.aiinterview.platform.application.mockinterview.dto;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * 模拟面试中间状态快照。
 */
public record MockInterviewStateSnapshot(
        String taskId,
        String userId,
        int attemptNo,
        String currentQuestionId,
        String currentRoundId,
        String waitingInputType,
        Instant waitingSince,
        Instant expiresAt,
        Map<String, Object> statePayload
) {
    public MockInterviewStateSnapshot {
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(currentQuestionId, "currentQuestionId 不能为空");
        Objects.requireNonNull(currentRoundId, "currentRoundId 不能为空");
        Objects.requireNonNull(waitingInputType, "waitingInputType 不能为空");
        Objects.requireNonNull(waitingSince, "waitingSince 不能为空");
        Objects.requireNonNull(expiresAt, "expiresAt 不能为空");
        Objects.requireNonNull(statePayload, "statePayload 不能为空");
    }
}
