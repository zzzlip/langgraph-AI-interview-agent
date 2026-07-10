package com.aiinterview.platform.application.mockinterview.command;

import java.util.Objects;

/**
 * 提交模拟面试回答命令。
 */
public record SubmitMockInterviewAnswerCommand(
        String userId,
        String taskId,
        String questionId,
        String roundId,
        String answerFileId,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public SubmitMockInterviewAnswerCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(questionId, "questionId 不能为空");
        Objects.requireNonNull(roundId, "roundId 不能为空");
        Objects.requireNonNull(answerFileId, "answerFileId 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
