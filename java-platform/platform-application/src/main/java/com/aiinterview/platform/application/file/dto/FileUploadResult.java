package com.aiinterview.platform.application.file.dto;

import java.time.Instant;

/**
 * 文件上传结果。
 */
public record FileUploadResult(
        String fileId,
        String fileName,
        String fileType,
        String filePurpose,
        long fileSize,
        String fileStatus,
        Instant createdAt
) {
}
