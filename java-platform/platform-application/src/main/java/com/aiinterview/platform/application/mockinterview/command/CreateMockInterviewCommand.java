package com.aiinterview.platform.application.mockinterview.command;

import com.aiinterview.platform.application.llm.LlmConfig;
import java.util.Objects;

/**
 * 创建模拟面试任务命令。
 */
public record CreateMockInterviewCommand(
        String userId,
        String resumeFileId,
        String targetJobTitle,
        String targetJobDescription,
        int questionCount,
        int maxFollowUpRounds,
        LlmConfig llmConfig,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public CreateMockInterviewCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(resumeFileId, "resumeFileId 不能为空");
        Objects.requireNonNull(targetJobTitle, "targetJobTitle 不能为空");
        Objects.requireNonNull(targetJobDescription, "targetJobDescription 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
