package com.aiinterview.platform.domain.file;

import java.time.Instant;
import java.util.Objects;

/**
 * 文件与任务的绑定关系。
 */
public final class FileBinding {
    private final String id;
    private final String fileId;
    private final String taskId;
    private final String userId;
    private final FileBindType bindType;
    private final int attemptNo;
    private final Instant createdAt;

    private FileBinding(
            String id,
            String fileId,
            String taskId,
            String userId,
            FileBindType bindType,
            int attemptNo,
            Instant createdAt
    ) {
        this.id = requireText(id, "绑定 ID 不能为空");
        this.fileId = requireText(fileId, "文件 ID 不能为空");
        this.taskId = requireText(taskId, "任务 ID 不能为空");
        this.userId = requireText(userId, "用户 ID 不能为空");
        this.bindType = Objects.requireNonNull(bindType, "绑定类型不能为空");
        this.attemptNo = requireNonNegative(attemptNo, "attemptNo 不能小于 0");
        this.createdAt = Objects.requireNonNull(createdAt, "创建时间不能为空");
    }

    public static FileBinding createInput(String id, String fileId, String taskId, String userId, Instant createdAt) {
        return new FileBinding(id, fileId, taskId, userId, FileBindType.INPUT, 0, createdAt);
    }

    public static FileBinding createInput(
            String id,
            String fileId,
            String taskId,
            String userId,
            int attemptNo,
            Instant createdAt
    ) {
        return new FileBinding(id, fileId, taskId, userId, FileBindType.INPUT, attemptNo, createdAt);
    }

    public static FileBinding createWorkerOutput(
            String id,
            String fileId,
            String taskId,
            String userId,
            FileBindType bindType,
            int attemptNo,
            Instant createdAt
    ) {
        if (bindType != FileBindType.OUTPUT && bindType != FileBindType.REPORT) {
            throw new IllegalArgumentException("Worker 产物只能绑定为 output 或 report");
        }
        if (attemptNo <= 0) {
            throw new IllegalArgumentException("Worker 产物 attemptNo 必须大于 0");
        }
        return new FileBinding(id, fileId, taskId, userId, bindType, attemptNo, createdAt);
    }

    public FileBindingUniqueKey uniqueKey() {
        return new FileBindingUniqueKey(fileId, taskId, bindType, attemptNo);
    }

    public String getId() {
        return id;
    }

    public String getFileId() {
        return fileId;
    }

    public String getTaskId() {
        return taskId;
    }

    public String getUserId() {
        return userId;
    }

    public FileBindType getBindType() {
        return bindType;
    }

    public int getAttemptNo() {
        return attemptNo;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    private static String requireText(String value, String message) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    private static int requireNonNegative(int value, String message) {
        if (value < 0) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    public record FileBindingUniqueKey(String fileId, String taskId, FileBindType bindType, int attemptNo) {
    }
}
