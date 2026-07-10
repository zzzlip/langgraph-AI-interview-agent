package com.aiinterview.platform.application.algorithmtest.dto;

/**
 * 算法测试提交检查发起结果。
 */
public record AlgorithmCheckRequestedResult(
        String taskId,
        String status,
        String statusLabel,
        int attempt
) {
}
