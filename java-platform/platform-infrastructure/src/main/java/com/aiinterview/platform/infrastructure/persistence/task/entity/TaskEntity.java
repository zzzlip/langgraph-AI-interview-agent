package com.aiinterview.platform.infrastructure.persistence.task.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 任务数据库实体。
 */
@TableName("`task`")
public class TaskEntity {

    @TableId
    private String id;
    @TableField("user_id")
    private String userId;
    @TableField("task_type")
    private String taskType;
    private String status;
    private String stage;
    private Integer progress;
    @TableField("input_summary")
    private String inputSummary;
    @TableField("result_reference_id")
    private String resultReferenceId;
    @TableField("current_attempt")
    private Integer currentAttempt;
    @TableField("retry_count")
    private Integer retryCount;
    @TableField("max_retry_count")
    private Integer maxRetryCount;
    private Boolean retryable;
    private Long version;
    @TableField("trace_id")
    private String traceId;
    @TableField("created_at")
    private Instant createdAt;
    @TableField("updated_at")
    private Instant updatedAt;
    @TableField("completed_at")
    private Instant completedAt;
    @TableField("cancelled_at")
    private Instant cancelledAt;
    @TableField("last_failed_at")
    private Instant lastFailedAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getTaskType() { return taskType; }
    public void setTaskType(String taskType) { this.taskType = taskType; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getStage() { return stage; }
    public void setStage(String stage) { this.stage = stage; }
    public Integer getProgress() { return progress; }
    public void setProgress(Integer progress) { this.progress = progress; }
    public String getInputSummary() { return inputSummary; }
    public void setInputSummary(String inputSummary) { this.inputSummary = inputSummary; }
    public String getResultReferenceId() { return resultReferenceId; }
    public void setResultReferenceId(String resultReferenceId) { this.resultReferenceId = resultReferenceId; }
    public Integer getCurrentAttempt() { return currentAttempt; }
    public void setCurrentAttempt(Integer currentAttempt) { this.currentAttempt = currentAttempt; }
    public Integer getRetryCount() { return retryCount; }
    public void setRetryCount(Integer retryCount) { this.retryCount = retryCount; }
    public Integer getMaxRetryCount() { return maxRetryCount; }
    public void setMaxRetryCount(Integer maxRetryCount) { this.maxRetryCount = maxRetryCount; }
    public Boolean getRetryable() { return retryable; }
    public void setRetryable(Boolean retryable) { this.retryable = retryable; }
    public Long getVersion() { return version; }
    public void setVersion(Long version) { this.version = version; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
    public Instant getCompletedAt() { return completedAt; }
    public void setCompletedAt(Instant completedAt) { this.completedAt = completedAt; }
    public Instant getCancelledAt() { return cancelledAt; }
    public void setCancelledAt(Instant cancelledAt) { this.cancelledAt = cancelledAt; }
    public Instant getLastFailedAt() { return lastFailedAt; }
    public void setLastFailedAt(Instant lastFailedAt) { this.lastFailedAt = lastFailedAt; }
}
