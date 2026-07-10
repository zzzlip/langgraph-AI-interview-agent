package com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 异步死信数据库实体。
 */
@TableName("async_dead_letter")
public class DeadLetterEntity {

    @TableId
    private String id;
    @TableField("source_type")
    private String sourceType;
    @TableField("source_id")
    private String sourceId;
    @TableField("task_id")
    private String taskId;
    @TableField("task_type")
    private String taskType;
    @TableField("attempt_no")
    private Integer attemptNo;
    @TableField("failure_stage")
    private String failureStage;
    @TableField("failure_reason")
    private String failureReason;
    private Boolean retryable;
    @TableField("manual_handled")
    private Boolean manualHandled;
    @TableField("trace_id")
    private String traceId;
    @TableField("created_at")
    private Instant createdAt;
    @TableField("updated_at")
    private Instant updatedAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSourceType() { return sourceType; }
    public void setSourceType(String sourceType) { this.sourceType = sourceType; }
    public String getSourceId() { return sourceId; }
    public void setSourceId(String sourceId) { this.sourceId = sourceId; }
    public String getTaskId() { return taskId; }
    public void setTaskId(String taskId) { this.taskId = taskId; }
    public String getTaskType() { return taskType; }
    public void setTaskType(String taskType) { this.taskType = taskType; }
    public Integer getAttemptNo() { return attemptNo; }
    public void setAttemptNo(Integer attemptNo) { this.attemptNo = attemptNo; }
    public String getFailureStage() { return failureStage; }
    public void setFailureStage(String failureStage) { this.failureStage = failureStage; }
    public String getFailureReason() { return failureReason; }
    public void setFailureReason(String failureReason) { this.failureReason = failureReason; }
    public Boolean getRetryable() { return retryable; }
    public void setRetryable(Boolean retryable) { this.retryable = retryable; }
    public Boolean getManualHandled() { return manualHandled; }
    public void setManualHandled(Boolean manualHandled) { this.manualHandled = manualHandled; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
