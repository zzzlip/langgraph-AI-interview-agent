package com.aiinterview.platform.application.file.dto;

import java.time.Instant;

/**
 * 文件详情结果。
 */
public record FileDetailResult(
        String fileId,
        String fileName,
        String fileType,
        String filePurpose,
        long fileSize,
        String fileStatus,
        Instant createdAt
) {
}
