package com.aiinterview.platform.domain.task;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * 任务历史事件。
 */
public class TaskEvent {

    private final String id;
    private final String taskId;
    private final Integer attemptNo;
    private final TaskEventType eventType;
    private final TaskStatus fromStatus;
    private final TaskStatus toStatus;
    private final String stage;
    private final Integer progress;
    private final String message;
    private final Map<String, Object> payload;
    private final TaskOperatorType operatorType;
    private final String operatorId;
    private final String traceId;
    private final Instant occurredAt;
    private final Instant createdAt;

    public TaskEvent(
            String id,
            String taskId,
            Integer attemptNo,
            TaskEventType eventType,
            TaskStatus fromStatus,
            TaskStatus toStatus,
            String stage,
            Integer progress,
            String message,
            Map<String, Object> payload,
            TaskOperatorType operatorType,
            String operatorId,
            String traceId,
            Instant occurredAt,
            Instant createdAt
    ) {
        validateMessage(message);
        validatePayload(payload);
        this.id = Objects.requireNonNull(id, "id 不能为空");
        this.taskId = Objects.requireNonNull(taskId, "taskId 不能为空");
        this.attemptNo = attemptNo;
        this.eventType = Objects.requireNonNull(eventType, "eventType 不能为空");
        this.fromStatus = fromStatus;
        this.toStatus = toStatus;
        this.stage = stage;
        this.progress = progress;
        this.message = message;
        this.payload = payload == null ? null : Map.copyOf(payload);
        this.operatorType = Objects.requireNonNull(operatorType, "operatorType 不能为空");
        this.operatorId = operatorId;
        this.traceId = Objects.requireNonNull(traceId, "traceId 不能为空");
        this.occurredAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
    }

    private static void validateMessage(String message) {
        if (message != null && message.length() > 500) {
            throw new IllegalArgumentException("message 最长 500 字符");
        }
    }

    private static void validatePayload(Map<String, Object> payload) {
        if (payload == null) {
            return;
        }
        Object schemaVersion = payload.get("schema_version");
        if (!(schemaVersion instanceof Integer)) {
            throw new IllegalArgumentException("payload 非空时 schema_version 必须为 integer");
        }
        if (maxDepth(payload) > 3) {
            throw new IllegalArgumentException("payload 嵌套层级不能超过 3 层");
        }
    }

    private static int maxDepth(Object value) {
        if (!(value instanceof Map<?, ?> map) || map.isEmpty()) {
            return 0;
        }
        int childDepth = 0;
        for (Object child : map.values()) {
            childDepth = Math.max(childDepth, maxDepth(child));
        }
        return 1 + childDepth;
    }

    public String getId() {
        return id;
    }

    public String getTaskId() {
        return taskId;
    }

    public Integer getAttemptNo() {
        return attemptNo;
    }

    public TaskEventType getEventType() {
        return eventType;
    }

    public TaskStatus getFromStatus() {
        return fromStatus;
    }

    public TaskStatus getToStatus() {
        return toStatus;
    }

    public String getStage() {
        return stage;
    }

    public Integer getProgress() {
        return progress;
    }

    public String getMessage() {
        return message;
    }

    public Map<String, Object> getPayload() {
        return payload;
    }

    public TaskOperatorType getOperatorType() {
        return operatorType;
    }

    public String getOperatorId() {
        return operatorId;
    }

    public String getTraceId() {
        return traceId;
    }

    public Instant getOccurredAt() {
        return occurredAt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
