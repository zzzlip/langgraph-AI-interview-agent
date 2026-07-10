package com.aiinterview.platform.application.result.dto;

/**
 * 结果关联文件条目。
 */
public record ResultFileItem(
        String fileId,
        String fileName,
        String fileType,
        long fileSize
) {
}
