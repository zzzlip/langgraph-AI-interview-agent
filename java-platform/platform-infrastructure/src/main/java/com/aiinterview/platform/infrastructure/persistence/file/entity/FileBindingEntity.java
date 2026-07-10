package com.aiinterview.platform.infrastructure.persistence.file.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 文件任务绑定数据库实体。
 */
@TableName("file_task_binding")
public class FileBindingEntity {
    @TableId
    private String id;
    @TableField("file_id")
    private String fileId;
    @TableField("task_id")
    private String taskId;
    @TableField("user_id")
    private String userId;
    @TableField("bind_type")
    private String bindType;
    @TableField("attempt_no")
    private Integer attemptNo;
    @TableField("created_at")
    private Instant createdAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getFileId() { return fileId; }
    public void setFileId(String fileId) { this.fileId = fileId; }
    public String getTaskId() { return taskId; }
    public void setTaskId(String taskId) { this.taskId = taskId; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getBindType() { return bindType; }
    public void setBindType(String bindType) { this.bindType = bindType; }
    public Integer getAttemptNo() { return attemptNo; }
    public void setAttemptNo(Integer attemptNo) { this.attemptNo = attemptNo; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
