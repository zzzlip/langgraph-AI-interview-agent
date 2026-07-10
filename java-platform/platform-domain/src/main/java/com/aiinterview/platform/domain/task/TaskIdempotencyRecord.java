package com.aiinterview.platform.domain.task;

import java.time.Instant;
import java.util.Objects;

/**
 * 任务幂等成功记录。
 */
public class TaskIdempotencyRecord {

    private final String id;
    private final String userId;
    private final String idempotencyKey;
    private final String operationType;
    private final String requestHash;
    private final String taskId;
    private final String responseSnapshot;
    private final Instant expiresAt;
    private final Instant createdAt;
    private final Instant updatedAt;

    public TaskIdempotencyRecord(
            String id,
            String userId,
            String idempotencyKey,
            String operationType,
            String requestHash,
            String taskId,
            String responseSnapshot,
            Instant expiresAt,
            Instant createdAt,
            Instant updatedAt
    ) {
        this.id = Objects.requireNonNull(id, "id 不能为空");
        this.userId = Objects.requireNonNull(userId, "userId 不能为空");
        this.idempotencyKey = Objects.requireNonNull(idempotencyKey, "idempotencyKey 不能为空");
        this.operationType = Objects.requireNonNull(operationType, "operationType 不能为空");
        this.requestHash = Objects.requireNonNull(requestHash, "requestHash 不能为空");
        this.taskId = Objects.requireNonNull(taskId, "taskId 不能为空");
        this.responseSnapshot = Objects.requireNonNull(responseSnapshot, "responseSnapshot 不能为空");
        this.expiresAt = Objects.requireNonNull(expiresAt, "expiresAt 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.updatedAt = Objects.requireNonNull(updatedAt, "updatedAt 不能为空");
    }

    public String getId() {
        return id;
    }

    public String getUserId() {
        return userId;
    }

    public String getIdempotencyKey() {
        return idempotencyKey;
    }

    public String getOperationType() {
        return operationType;
    }

    public String getRequestHash() {
        return requestHash;
    }

    public String getTaskId() {
        return taskId;
    }

    public String getResponseSnapshot() {
        return responseSnapshot;
    }

    public Instant getExpiresAt() {
        return expiresAt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }
}
