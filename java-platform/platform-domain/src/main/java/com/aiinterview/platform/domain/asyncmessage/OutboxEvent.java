package com.aiinterview.platform.domain.asyncmessage;

import java.time.Instant;
import java.util.Objects;

/**
 * 任务命令 Outbox 事件。
 */
public class OutboxEvent {

    private final String id;
    private final String eventId;
    private final String aggregateType;
    private final String aggregateId;
    private final String taskId;
    private final String taskType;
    private final int attemptNo;
    private final String eventType;
    private final String routingKey;
    private final String payload;
    private final AsyncDeliveryStatus status;
    private final int retryCount;
    private final int maxRetryCount;
    private final Instant nextRetryAt;
    private final String lastError;
    private final String traceId;
    private final Instant createdAt;
    private final Instant updatedAt;
    private final Instant deliveredAt;

    public OutboxEvent(
            String id,
            String eventId,
            String aggregateType,
            String aggregateId,
            String taskId,
            String taskType,
            int attemptNo,
            String eventType,
            String routingKey,
            String payload,
            AsyncDeliveryStatus status,
            int retryCount,
            int maxRetryCount,
            Instant nextRetryAt,
            String lastError,
            String traceId,
            Instant createdAt,
            Instant updatedAt,
            Instant deliveredAt
    ) {
        if (attemptNo < 1) {
            throw new IllegalArgumentException("attemptNo 必须从 1 开始");
        }
        if (retryCount < 0) {
            throw new IllegalArgumentException("retryCount 不能小于 0");
        }
        if (maxRetryCount < 1) {
            throw new IllegalArgumentException("maxRetryCount 必须大于 0");
        }
        this.id = requireText(id, "id 不能为空");
        this.eventId = requireText(eventId, "eventId 不能为空");
        this.aggregateType = requireText(aggregateType, "aggregateType 不能为空");
        this.aggregateId = requireText(aggregateId, "aggregateId 不能为空");
        this.taskId = requireText(taskId, "taskId 不能为空");
        this.taskType = requireText(taskType, "taskType 不能为空");
        this.attemptNo = attemptNo;
        this.eventType = requireText(eventType, "eventType 不能为空");
        this.routingKey = requireText(routingKey, "routingKey 不能为空");
        this.payload = requireText(payload, "payload 不能为空");
        this.status = Objects.requireNonNull(status, "status 不能为空");
        this.retryCount = retryCount;
        this.maxRetryCount = maxRetryCount;
        this.nextRetryAt = nextRetryAt;
        this.lastError = lastError;
        this.traceId = requireText(traceId, "traceId 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.updatedAt = Objects.requireNonNull(updatedAt, "updatedAt 不能为空");
        this.deliveredAt = deliveredAt;
    }

    public boolean canPublish(Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status == AsyncDeliveryStatus.PENDING) {
            return true;
        }
        return status == AsyncDeliveryStatus.FAILED
                && retryCount < maxRetryCount
                && (nextRetryAt == null || !nextRetryAt.isAfter(now));
    }

    public OutboxEvent markDelivering(Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status != AsyncDeliveryStatus.PENDING && status != AsyncDeliveryStatus.FAILED) {
            throw new IllegalStateException("当前 Outbox 状态不允许投递");
        }
        if (retryCount >= maxRetryCount) {
            throw new IllegalStateException("Outbox 投递次数已达上限");
        }
        return copy(AsyncDeliveryStatus.DELIVERING, retryCount + 1, nextRetryAt, null, now, deliveredAt);
    }

    public OutboxEvent markDelivered(Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status != AsyncDeliveryStatus.DELIVERING) {
            throw new IllegalStateException("当前 Outbox 状态不允许标记已投递");
        }
        return copy(AsyncDeliveryStatus.DELIVERED, retryCount, nextRetryAt, null, now, now);
    }

    public OutboxEvent markFailed(String error, Instant nextRetryAt, Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status != AsyncDeliveryStatus.DELIVERING) {
            throw new IllegalStateException("当前 Outbox 状态不允许标记失败");
        }
        return copy(AsyncDeliveryStatus.FAILED, retryCount, nextRetryAt, error, now, deliveredAt);
    }

    public OutboxEvent markDead(String error, Instant now) {
        Objects.requireNonNull(now, "now 不能为空");
        if (status.isFinal()) {
            throw new IllegalStateException("终态 Outbox 不能进入死信");
        }
        return copy(AsyncDeliveryStatus.DEAD, retryCount, nextRetryAt, error, now, deliveredAt);
    }

    private OutboxEvent copy(
            AsyncDeliveryStatus nextStatus,
            int nextRetryCount,
            Instant nextRetryAt,
            String nextLastError,
            Instant nextUpdatedAt,
            Instant nextDeliveredAt
    ) {
        return new OutboxEvent(id, eventId, aggregateType, aggregateId, taskId, taskType, attemptNo, eventType,
                routingKey, payload, nextStatus, nextRetryCount, maxRetryCount, nextRetryAt, nextLastError,
                traceId, createdAt, nextUpdatedAt, nextDeliveredAt);
    }

    private static String requireText(String value, String message) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    public String getId() {
        return id;
    }

    public String getEventId() {
        return eventId;
    }

    public String getAggregateType() {
        return aggregateType;
    }

    public String getAggregateId() {
        return aggregateId;
    }

    public String getTaskId() {
        return taskId;
    }

    public String getTaskType() {
        return taskType;
    }

    public int getAttemptNo() {
        return attemptNo;
    }

    public String getEventType() {
        return eventType;
    }

    public String getRoutingKey() {
        return routingKey;
    }

    public String getPayload() {
        return payload;
    }

    public AsyncDeliveryStatus getStatus() {
        return status;
    }

    public int getRetryCount() {
        return retryCount;
    }

    public int getMaxRetryCount() {
        return maxRetryCount;
    }

    public Instant getNextRetryAt() {
        return nextRetryAt;
    }

    public String getLastError() {
        return lastError;
    }

    public String getTraceId() {
        return traceId;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public Instant getDeliveredAt() {
        return deliveredAt;
    }
}
