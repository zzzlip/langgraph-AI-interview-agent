package com.aiinterview.platform.application.file.dto;

import java.time.Instant;

/**
 * 管理员文件状态摘要。
 */
public record AdminFileStatusItem(
        String fileId,
        String taskId,
        String fileType,
        String filePurpose,
        String fileStatus,
        String failureReason,
        Instant createdAt,
        Instant updatedAt
) {
}
