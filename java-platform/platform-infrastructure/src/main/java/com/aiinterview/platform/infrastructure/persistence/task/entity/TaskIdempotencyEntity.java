package com.aiinterview.platform.infrastructure.persistence.task.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 任务幂等记录数据库实体。
 */
@TableName("`task_idempotency_record`")
public class TaskIdempotencyEntity {

    @TableId
    private String id;
    @TableField("user_id")
    private String userId;
    @TableField("idempotency_key")
    private String idempotencyKey;
    @TableField("operation_type")
    private String operationType;
    @TableField("request_hash")
    private String requestHash;
    @TableField("task_id")
    private String taskId;
    @TableField("response_snapshot")
    private String responseSnapshot;
    @TableField("expires_at")
    private Instant expiresAt;
    @TableField("created_at")
    private Instant createdAt;
    @TableField("updated_at")
    private Instant updatedAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getIdempotencyKey() { return idempotencyKey; }
    public void setIdempotencyKey(String idempotencyKey) { this.idempotencyKey = idempotencyKey; }
    public String getOperationType() { return operationType; }
    public void setOperationType(String operationType) { this.operationType = operationType; }
    public String getRequestHash() { return requestHash; }
    public void setRequestHash(String requestHash) { this.requestHash = requestHash; }
    public String getTaskId() { return taskId; }
    public void setTaskId(String taskId) { this.taskId = taskId; }
    public String getResponseSnapshot() { return responseSnapshot; }
    public void setResponseSnapshot(String responseSnapshot) { this.responseSnapshot = responseSnapshot; }
    public Instant getExpiresAt() { return expiresAt; }
    public void setExpiresAt(Instant expiresAt) { this.expiresAt = expiresAt; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
