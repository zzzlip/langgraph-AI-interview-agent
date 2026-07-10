package com.aiinterview.platform.application.resumeoptimization.dto;

/**
 * 简历优化输入文件元信息。
 */
public record ResumeOptimizationInputFile(
        String fileId,
        String originalName,
        String objectKey,
        String fileType,
        String filePurpose
) {
}
