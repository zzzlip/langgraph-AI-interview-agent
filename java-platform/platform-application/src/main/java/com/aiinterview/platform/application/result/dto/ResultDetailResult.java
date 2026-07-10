package com.aiinterview.platform.application.result.dto;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 任务详情中的结果扩展。
 */
public record ResultDetailResult(
        String resultId,
        String resultType,
        Map<String, Object> summary,
        List<ResultFileItem> reportFiles,
        List<ResultFileItem> outputFiles,
        String resultStatus,
        Instant createdAt,
        Instant completedAt
) {
    public ResultDetailResult {
        summary = Map.copyOf(Objects.requireNonNull(summary, "summary 不能为空"));
        reportFiles = List.copyOf(Objects.requireNonNull(reportFiles, "reportFiles 不能为空"));
        outputFiles = List.copyOf(Objects.requireNonNull(outputFiles, "outputFiles 不能为空"));
    }
}
