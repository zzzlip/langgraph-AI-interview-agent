package com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * Inbox 事件数据库实体。
 */
@TableName("inbox_event")
public class InboxEventEntity {

    @TableId
    private String id;
    @TableField("event_id")
    private String eventId;
    @TableField("event_type")
    private String eventType;
    @TableField("task_id")
    private String taskId;
    @TableField("task_type")
    private String taskType;
    @TableField("attempt_no")
    private Integer attemptNo;
    @TableField("feedback_type")
    private String feedbackType;
    private String status;
    @TableField("payload_hash")
    private String payloadHash;
    @TableField("payload_summary")
    private String payloadSummary;
    @TableField("retry_count")
    private Integer retryCount;
    @TableField("max_retry_count")
    private Integer maxRetryCount;
    @TableField("last_error")
    private String lastError;
    @TableField("trace_id")
    private String traceId;
    @TableField("received_at")
    private Instant receivedAt;
    @TableField("processed_at")
    private Instant processedAt;
    @TableField("created_at")
    private Instant createdAt;
    @TableField("updated_at")
    private Instant updatedAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getEventId() { return eventId; }
    public void setEventId(String eventId) { this.eventId = eventId; }
    public String getEventType() { return eventType; }
    public void setEventType(String eventType) { this.eventType = eventType; }
    public String getTaskId() { return taskId; }
    public void setTaskId(String taskId) { this.taskId = taskId; }
    public String getTaskType() { return taskType; }
    public void setTaskType(String taskType) { this.taskType = taskType; }
    public Integer getAttemptNo() { return attemptNo; }
    public void setAttemptNo(Integer attemptNo) { this.attemptNo = attemptNo; }
    public String getFeedbackType() { return feedbackType; }
    public void setFeedbackType(String feedbackType) { this.feedbackType = feedbackType; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getPayloadHash() { return payloadHash; }
    public void setPayloadHash(String payloadHash) { this.payloadHash = payloadHash; }
    public String getPayloadSummary() { return payloadSummary; }
    public void setPayloadSummary(String payloadSummary) { this.payloadSummary = payloadSummary; }
    public Integer getRetryCount() { return retryCount; }
    public void setRetryCount(Integer retryCount) { this.retryCount = retryCount; }
    public Integer getMaxRetryCount() { return maxRetryCount; }
    public void setMaxRetryCount(Integer maxRetryCount) { this.maxRetryCount = maxRetryCount; }
    public String getLastError() { return lastError; }
    public void setLastError(String lastError) { this.lastError = lastError; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }
    public Instant getReceivedAt() { return receivedAt; }
    public void setReceivedAt(Instant receivedAt) { this.receivedAt = receivedAt; }
    public Instant getProcessedAt() { return processedAt; }
    public void setProcessedAt(Instant processedAt) { this.processedAt = processedAt; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
