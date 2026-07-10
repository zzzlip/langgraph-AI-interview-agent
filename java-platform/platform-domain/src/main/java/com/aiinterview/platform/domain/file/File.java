package com.aiinterview.platform.domain.file;

import java.time.Instant;
import java.util.Objects;

/**
 * 文件聚合根。
 */
public final class File {
    private final String id;
    private final String userId;
    private final String originalName;
    private final String storedFilename;
    private final FileType fileType;
    private final FilePurpose filePurpose;
    private final long fileSize;
    private final String mimeType;
    private final String sourceDeviceId;
    private final String objectKey;
    private final Instant createdAt;
    private FileStatus fileStatus;
    private String checksum;
    private String failureReason;
    private Instant updatedAt;

    private File(
            String id,
            String userId,
            String originalName,
            String storedFilename,
            FileType fileType,
            FilePurpose filePurpose,
            long fileSize,
            String mimeType,
            String sourceDeviceId,
            String objectKey,
            Instant createdAt
    ) {
        this.id = requireText(id, "文件 ID 不能为空");
        this.userId = requireText(userId, "用户 ID 不能为空");
        this.originalName = requireText(originalName, "原始文件名不能为空");
        this.storedFilename = requireText(storedFilename, "存储文件名不能为空");
        this.fileType = Objects.requireNonNull(fileType, "文件类型不能为空");
        this.filePurpose = Objects.requireNonNull(filePurpose, "文件用途不能为空");
        this.fileSize = requirePositive(fileSize, "文件大小必须大于 0");
        this.mimeType = requireText(mimeType, "MIME 类型不能为空");
        this.sourceDeviceId = normalizeNullableText(sourceDeviceId);
        this.objectKey = requireText(objectKey, "对象存储 key 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "创建时间不能为空");
        this.updatedAt = createdAt;
        this.fileStatus = FileStatus.UPLOADING;
        validatePurposeAndType(filePurpose, fileType);
    }

    public static File createUploading(
            String id,
            String userId,
            String originalName,
            String storedFilename,
            FileType fileType,
            FilePurpose filePurpose,
            long fileSize,
            String mimeType,
            String objectKey,
            Instant createdAt
    ) {
        return new File(id, userId, originalName, storedFilename, fileType, filePurpose, fileSize, mimeType, null, objectKey, createdAt);
    }

    public static File createUploading(
            String id,
            String userId,
            String originalName,
            String storedFilename,
            FileType fileType,
            FilePurpose filePurpose,
            long fileSize,
            String mimeType,
            String sourceDeviceId,
            String objectKey,
            Instant createdAt
    ) {
        return new File(id, userId, originalName, storedFilename, fileType, filePurpose, fileSize, mimeType, sourceDeviceId, objectKey, createdAt);
    }

    public void markAvailable(String checksum, Instant updatedAt) {
        ensureMutable();
        this.checksum = checksum;
        this.fileStatus = FileStatus.AVAILABLE;
        this.updatedAt = Objects.requireNonNull(updatedAt, "更新时间不能为空");
    }

    public void markFailed(String failureReason, Instant updatedAt) {
        ensureMutable();
        this.failureReason = requireText(failureReason, "失败原因不能为空");
        this.fileStatus = FileStatus.FAILED;
        this.updatedAt = Objects.requireNonNull(updatedAt, "更新时间不能为空");
    }

    public boolean canBindAsInput() {
        return fileStatus == FileStatus.AVAILABLE && filePurpose.isUserUpload();
    }

    public void assertOwnedBy(String currentUserId) {
        if (!userId.equals(currentUserId)) {
            throw new FileOwnershipException("文件不存在或无权访问");
        }
    }

    public String getId() {
        return id;
    }

    public String getUserId() {
        return userId;
    }

    public String getOriginalName() {
        return originalName;
    }

    public String getStoredFilename() {
        return storedFilename;
    }

    public FileType getFileType() {
        return fileType;
    }

    public FilePurpose getFilePurpose() {
        return filePurpose;
    }

    public long getFileSize() {
        return fileSize;
    }

    public String getMimeType() {
        return mimeType;
    }

    public String getSourceDeviceId() {
        return sourceDeviceId;
    }

    public String getObjectKey() {
        return objectKey;
    }

    public FileStatus getFileStatus() {
        return fileStatus;
    }

    public String getChecksum() {
        return checksum;
    }

    public String getFailureReason() {
        return failureReason;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    private void ensureMutable() {
        if (fileStatus.isFinal()) {
            throw new IllegalStateException("终态文件不可变更");
        }
    }

    private static void validatePurposeAndType(FilePurpose purpose, FileType type) {
        if (!purpose.allows(type)) {
            throw new IllegalArgumentException("文件类型与用途不匹配");
        }
    }

    private static String requireText(String value, String message) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    private static String normalizeNullableText(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return value;
    }

    private static long requirePositive(long value, String message) {
        if (value <= 0) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }
}
