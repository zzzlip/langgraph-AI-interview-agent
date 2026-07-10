package com.aiinterview.platform.infrastructure.persistence.file.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 文件数据库实体。
 */
@TableName("`file`")
public class FileEntity {
    @TableId
    private String id;
    @TableField("user_id")
    private String userId;
    @TableField("original_name")
    private String originalName;
    @TableField("stored_filename")
    private String storedFilename;
    @TableField("file_type")
    private String fileType;
    @TableField("file_purpose")
    private String filePurpose;
    @TableField("file_size")
    private Long fileSize;
    @TableField("mime_type")
    private String mimeType;
    @TableField("source_device_id")
    private String sourceDeviceId;
    @TableField("file_status")
    private String fileStatus;
    @TableField("object_key")
    private String objectKey;
    private String checksum;
    @TableField("failure_reason")
    private String failureReason;
    @TableField("created_at")
    private Instant createdAt;
    @TableField("updated_at")
    private Instant updatedAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getOriginalName() { return originalName; }
    public void setOriginalName(String originalName) { this.originalName = originalName; }
    public String getStoredFilename() { return storedFilename; }
    public void setStoredFilename(String storedFilename) { this.storedFilename = storedFilename; }
    public String getFileType() { return fileType; }
    public void setFileType(String fileType) { this.fileType = fileType; }
    public String getFilePurpose() { return filePurpose; }
    public void setFilePurpose(String filePurpose) { this.filePurpose = filePurpose; }
    public Long getFileSize() { return fileSize; }
    public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
    public String getMimeType() { return mimeType; }
    public void setMimeType(String mimeType) { this.mimeType = mimeType; }
    public String getSourceDeviceId() { return sourceDeviceId; }
    public void setSourceDeviceId(String sourceDeviceId) { this.sourceDeviceId = sourceDeviceId; }
    public String getFileStatus() { return fileStatus; }
    public void setFileStatus(String fileStatus) { this.fileStatus = fileStatus; }
    public String getObjectKey() { return objectKey; }
    public void setObjectKey(String objectKey) { this.objectKey = objectKey; }
    public String getChecksum() { return checksum; }
    public void setChecksum(String checksum) { this.checksum = checksum; }
    public String getFailureReason() { return failureReason; }
    public void setFailureReason(String failureReason) { this.failureReason = failureReason; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
