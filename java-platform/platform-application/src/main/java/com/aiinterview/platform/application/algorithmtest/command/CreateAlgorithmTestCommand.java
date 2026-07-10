package com.aiinterview.platform.application.algorithmtest.command;

import com.aiinterview.platform.application.llm.LlmConfig;
import java.util.Objects;

/**
 * 创建算法测试任务命令。
 */
public record CreateAlgorithmTestCommand(
        String userId,
        String targetJobTitle,
        String targetJobDescription,
        String codeforcesHandle,
        LlmConfig llmConfig,
        String idempotencyKey,
        String requestHash,
        String traceId
) {
    public CreateAlgorithmTestCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(targetJobTitle, "targetJobTitle 不能为空");
        Objects.requireNonNull(targetJobDescription, "targetJobDescription 不能为空");
        Objects.requireNonNull(codeforcesHandle, "codeforcesHandle 不能为空");
        Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        Objects.requireNonNull(requestHash, "requestHash 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
    }
}
