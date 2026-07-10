package com.aiinterview.platform.application.file.dto;

import java.time.Instant;

/**
 * 任务关联文件条目。
 */
public record FileTaskFileItem(
        String fileId,
        String fileName,
        String fileType,
        String filePurpose,
        long fileSize,
        String bindType,
        Instant createdAt
) {
}
