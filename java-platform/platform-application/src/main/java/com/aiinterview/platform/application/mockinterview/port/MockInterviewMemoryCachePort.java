package com.aiinterview.platform.application.mockinterview.port;

import java.util.Map;
import java.util.Optional;

/**
 * 模拟面试任务内记忆 Redis 热缓存端口。
 */
public interface MockInterviewMemoryCachePort {

    void refresh(
            String taskId,
            int attemptNo,
            long stateVersion,
            String currentQuestionId,
            String currentRoundId,
            Object taskMemory
    );

    Optional<Map<String, Object>> findValid(String taskId, int attemptNo, long expectedStateVersion);
}
