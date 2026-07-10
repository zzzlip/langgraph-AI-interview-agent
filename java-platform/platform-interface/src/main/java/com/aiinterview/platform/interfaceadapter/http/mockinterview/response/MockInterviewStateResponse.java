package com.aiinterview.platform.interfaceadapter.http.mockinterview.response;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCurrentState;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 模拟面试当前状态响应。
 */
public record MockInterviewStateResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("attempt_no") int attemptNo,
        @JsonProperty("current_question_id") String currentQuestionId,
        @JsonProperty("current_round_id") String currentRoundId,
        @JsonProperty("waiting_input_type") String waitingInputType,
        @JsonProperty("state_payload") Map<String, Object> statePayload
) {
    public static MockInterviewStateResponse from(MockInterviewCurrentState state) {
        return new MockInterviewStateResponse(
                state.taskId(),
                state.attemptNo(),
                state.currentQuestionId(),
                state.currentRoundId(),
                state.waitingInputType(),
                sanitizeMap(state.statePayload())
        );
    }

    private static Map<String, Object> sanitizeMap(Map<String, Object> source) {
        Map<String, Object> sanitized = new LinkedHashMap<>();
        source.forEach((key, value) -> {
            if (!sensitiveKey(key)) {
                sanitized.put(key, sanitizeValue(value));
            }
        });
        return sanitized;
    }

    private static Object sanitizeValue(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> nested = new LinkedHashMap<>();
            map.forEach((key, item) -> {
                String textKey = String.valueOf(key);
                if (!sensitiveKey(textKey)) {
                    nested.put(textKey, sanitizeValue(item));
                }
            });
            return nested;
        }
        if (value instanceof List<?> list) {
            List<Object> nested = new ArrayList<>();
            list.forEach(item -> nested.add(sanitizeValue(item)));
            return nested;
        }
        return value;
    }

    private static boolean sensitiveKey(String key) {
        String normalized = key == null ? "" : key.toLowerCase(Locale.ROOT);
        return normalized.contains("object_key")
                || normalized.contains("api_key")
                || normalized.contains("llm_config")
                || normalized.contains("task_memory")
                || normalized.contains("prompt")
                || normalized.contains("model_response")
                || normalized.contains("transcript")
                || normalized.contains("token")
                || normalized.contains("cookie");
    }
}
