package com.aiinterview.platform.application.mockinterview.dto;

/**
 * 模拟面试回答文件安全元信息。
 */
public record MockInterviewAnswerFile(
        String fileId,
        String objectKey,
        String fileType,
        String filePurpose,
        String mediaType,
        String mimeType
) {
}
