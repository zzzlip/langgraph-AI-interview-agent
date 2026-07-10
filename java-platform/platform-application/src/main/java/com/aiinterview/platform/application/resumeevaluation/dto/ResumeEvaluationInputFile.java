package com.aiinterview.platform.application.resumeevaluation.dto;

/**
 * 简历评估输入文件元信息。
 */
public record ResumeEvaluationInputFile(
        String fileId,
        String originalName,
        String objectKey,
        String fileType,
        String filePurpose
) {
}
