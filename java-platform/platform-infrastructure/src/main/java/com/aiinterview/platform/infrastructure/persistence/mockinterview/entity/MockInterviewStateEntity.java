package com.aiinterview.platform.infrastructure.persistence.mockinterview.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 模拟面试中间状态数据库实体。
 */
@TableName("mock_interview_state")
public class MockInterviewStateEntity {
    @TableId
    private String id;
    @TableField("task_id")
    private String taskId;
    @TableField("user_id")
    private String userId;
    @TableField("attempt_no")
    private Integer attemptNo;
    @TableField("current_question_id")
    private String currentQuestionId;
    @TableField("current_round_id")
    private String currentRoundId;
    @TableField("waiting_input_type")
    private String waitingInputType;
    @TableField("input_expired")
    private Boolean inputExpired;
    @TableField("waiting_since")
    private Instant waitingSince;
    @TableField("expires_at")
    private Instant expiresAt;
    @TableField("state_payload")
    private String statePayload;
    private Long version;
    @TableField("created_at")
    private Instant createdAt;
    @TableField("updated_at")
    private Instant updatedAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getTaskId() { return taskId; }
    public void setTaskId(String taskId) { this.taskId = taskId; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public Integer getAttemptNo() { return attemptNo; }
    public void setAttemptNo(Integer attemptNo) { this.attemptNo = attemptNo; }
    public String getCurrentQuestionId() { return currentQuestionId; }
    public void setCurrentQuestionId(String currentQuestionId) { this.currentQuestionId = currentQuestionId; }
    public String getCurrentRoundId() { return currentRoundId; }
    public void setCurrentRoundId(String currentRoundId) { this.currentRoundId = currentRoundId; }
    public String getWaitingInputType() { return waitingInputType; }
    public void setWaitingInputType(String waitingInputType) { this.waitingInputType = waitingInputType; }
    public Boolean getInputExpired() { return inputExpired; }
    public void setInputExpired(Boolean inputExpired) { this.inputExpired = inputExpired; }
    public Instant getWaitingSince() { return waitingSince; }
    public void setWaitingSince(Instant waitingSince) { this.waitingSince = waitingSince; }
    public Instant getExpiresAt() { return expiresAt; }
    public void setExpiresAt(Instant expiresAt) { this.expiresAt = expiresAt; }
    public String getStatePayload() { return statePayload; }
    public void setStatePayload(String statePayload) { this.statePayload = statePayload; }
    public Long getVersion() { return version; }
    public void setVersion(Long version) { this.version = version; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
