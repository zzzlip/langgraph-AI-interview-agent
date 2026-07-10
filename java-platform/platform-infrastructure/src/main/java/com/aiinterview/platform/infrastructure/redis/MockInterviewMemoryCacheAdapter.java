package com.aiinterview.platform.infrastructure.redis;

import com.aiinterview.platform.application.mockinterview.port.MockInterviewMemoryCachePort;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

/**
 * 模拟面试任务内记忆 Redis 热缓存。
 */
@Component
public class MockInterviewMemoryCacheAdapter implements MockInterviewMemoryCachePort {

    private static final TypeReference<Map<String, Object>> PAYLOAD_TYPE = new TypeReference<>() {
    };
    private static final Duration TTL = Duration.ofHours(96);

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;

    public MockInterviewMemoryCacheAdapter(StringRedisTemplate redisTemplate, ObjectMapper objectMapper) {
        this.redisTemplate = redisTemplate;
        this.objectMapper = objectMapper;
    }

    @Override
    public void refresh(
            String taskId,
            int attemptNo,
            long stateVersion,
            String currentQuestionId,
            String currentRoundId,
            Object taskMemory
    ) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("task_id", taskId);
        payload.put("attempt", attemptNo);
        payload.put("state_version", stateVersion);
        payload.put("task_memory", taskMemory);
        payload.put("current_question_id", currentQuestionId);
        payload.put("current_round_id", currentRoundId);
        payload.put("updated_at", Instant.now().toString());
        try {
            redisTemplate.opsForValue().set(key(taskId, attemptNo), objectMapper.writeValueAsString(payload), TTL);
        } catch (Exception ex) {
            throw new IllegalStateException("模拟面试记忆缓存写入失败", ex);
        }
    }

    @Override
    public Optional<Map<String, Object>> findValid(String taskId, int attemptNo, long expectedStateVersion) {
        String value = redisTemplate.opsForValue().get(key(taskId, attemptNo));
        if (value == null || value.isBlank()) {
            return Optional.empty();
        }
        try {
            Map<String, Object> payload = objectMapper.readValue(value, PAYLOAD_TYPE);
            Object version = payload.get("state_version");
            if (!(version instanceof Number number) || number.longValue() != expectedStateVersion) {
                return Optional.empty();
            }
            return Optional.of(payload);
        } catch (Exception ex) {
            return Optional.empty();
        }
    }

    private static String key(String taskId, int attemptNo) {
        return "mock_interview:memory:" + taskId + ":" + attemptNo;
    }
}
