package com.aiinterview.platform.application.resumeoptimization.command;

import com.aiinterview.platform.application.llm.LlmConfig;
import java.util.Objects;

/**
 * 创建简历优化任务命令。
 */
public record CreateResumeOptimizationCommand(
        String userId,
        String resumeFileId,
        String avatarFileId,
        String targetJobTitle,
        String targetJobDescription,
        LlmConfig llmConfig,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public CreateResumeOptimizationCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(resumeFileId, "resumeFileId 不能为空");
        Objects.requireNonNull(targetJobTitle, "targetJobTitle 不能为空");
        Objects.requireNonNull(targetJobDescription, "targetJobDescription 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
