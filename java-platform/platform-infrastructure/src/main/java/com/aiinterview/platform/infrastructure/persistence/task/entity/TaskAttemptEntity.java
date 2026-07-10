package com.aiinterview.platform.infrastructure.persistence.task.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 任务执行尝试数据库实体。
 */
@TableName("`task_attempt`")
public class TaskAttemptEntity {

    @TableId
    private String id;
    @TableField("task_id")
    private String taskId;
    @TableField("attempt_no")
    private Integer attemptNo;
    private String status;
    @TableField("started_at")
    private Instant startedAt;
    @TableField("finished_at")
    private Instant finishedAt;
    @TableField("failed_at")
    private Instant failedAt;
    @TableField("failure_stage")
    private String failureStage;
    @TableField("failure_code")
    private String failureCode;
    @TableField("user_message")
    private String userMessage;
    @TableField("admin_summary")
    private String adminSummary;
    @TableField("created_at")
    private Instant createdAt;
    @TableField("updated_at")
    private Instant updatedAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getTaskId() { return taskId; }
    public void setTaskId(String taskId) { this.taskId = taskId; }
    public Integer getAttemptNo() { return attemptNo; }
    public void setAttemptNo(Integer attemptNo) { this.attemptNo = attemptNo; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public Instant getStartedAt() { return startedAt; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }
    public Instant getFinishedAt() { return finishedAt; }
    public void setFinishedAt(Instant finishedAt) { this.finishedAt = finishedAt; }
    public Instant getFailedAt() { return failedAt; }
    public void setFailedAt(Instant failedAt) { this.failedAt = failedAt; }
    public String getFailureStage() { return failureStage; }
    public void setFailureStage(String failureStage) { this.failureStage = failureStage; }
    public String getFailureCode() { return failureCode; }
    public void setFailureCode(String failureCode) { this.failureCode = failureCode; }
    public String getUserMessage() { return userMessage; }
    public void setUserMessage(String userMessage) { this.userMessage = userMessage; }
    public String getAdminSummary() { return adminSummary; }
    public void setAdminSummary(String adminSummary) { this.adminSummary = adminSummary; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
