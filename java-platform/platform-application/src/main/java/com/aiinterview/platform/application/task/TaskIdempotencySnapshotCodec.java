package com.aiinterview.platform.application.task;

import com.aiinterview.platform.application.task.dto.TaskCreatedResult;
import com.aiinterview.platform.application.task.dto.TaskStatusResult;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.LinkedHashMap;
import java.util.Map;

final class TaskIdempotencySnapshotCodec {

    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private TaskIdempotencySnapshotCodec() {
    }

    static String toCreatedSnapshot(TaskCreatedResult result) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("task_id", result.taskId());
        snapshot.put("status", result.status());
        return writeSnapshot(snapshot);
    }

    static TaskCreatedResult parseCreatedSnapshot(String responseSnapshot) {
        JsonNode root = readSnapshot(responseSnapshot);
        return new TaskCreatedResult(requiredText(root, "task_id"), requiredText(root, "status"));
    }

    static String toStatusSnapshot(TaskStatusResult result) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("task_id", result.taskId());
        snapshot.put("status", result.status());
        snapshot.put("attempt", result.attempt());
        snapshot.put("retry_count", result.retryCount());
        snapshot.put("max_retry_count", result.maxRetryCount());
        return writeSnapshot(snapshot);
    }

    static TaskStatusResult parseStatusSnapshot(String responseSnapshot) {
        JsonNode root = readSnapshot(responseSnapshot);
        return new TaskStatusResult(
                requiredText(root, "task_id"),
                requiredText(root, "status"),
                null,
                requiredInt(root, "attempt"),
                requiredInt(root, "retry_count"),
                requiredInt(root, "max_retry_count")
        );
    }

    private static String writeSnapshot(Map<String, Object> snapshot) {
        try {
            return OBJECT_MAPPER.writeValueAsString(snapshot);
        } catch (JsonProcessingException ex) {
            throw invalidSnapshot(ex);
        }
    }

    private static JsonNode readSnapshot(String responseSnapshot) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseSnapshot);
            if (root == null || !root.isObject()) {
                throw invalidSnapshot(null);
            }
            return root;
        } catch (JsonProcessingException | IllegalArgumentException ex) {
            throw invalidSnapshot(ex);
        }
    }

    private static String requiredText(JsonNode root, String field) {
        JsonNode value = root.get(field);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw invalidSnapshot(null);
        }
        return value.asText();
    }

    private static int requiredInt(JsonNode root, String field) {
        JsonNode value = root.get(field);
        if (value == null || !value.canConvertToInt()) {
            throw invalidSnapshot(null);
        }
        return value.asInt();
    }

    private static TaskException invalidSnapshot(Throwable cause) {
        if (cause == null) {
            return new TaskException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照格式非法");
        }
        return new TaskException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照格式非法", cause);
    }
}
