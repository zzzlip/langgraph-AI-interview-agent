package com.aiinterview.platform.application.mockinterview.dto;

/**
 * 模拟面试简历文件元信息。
 */
public record MockInterviewResumeFile(
        String fileId,
        String originalName,
        String objectKey,
        String fileType,
        String filePurpose,
        String mimeType
) {
}
