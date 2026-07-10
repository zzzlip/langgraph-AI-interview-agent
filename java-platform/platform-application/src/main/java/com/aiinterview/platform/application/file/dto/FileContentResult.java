package com.aiinterview.platform.application.file.dto;

import java.io.InputStream;

/**
 * 文件内容读取结果。
 */
public record FileContentResult(
        String fileId,
        String fileName,
        String mimeType,
        long fileSize,
        String cacheControl,
        boolean inline,
        InputStream content
) {
}
