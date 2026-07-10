package com.aiinterview.platform.domain.result;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 任务结构化结果。
 */
public final class TaskResult {
    private final String id;
    private final String taskId;
    private final int attemptNo;
    private final ResultType resultType;
    private final Map<String, Object> summary;
    private final ResultStatus resultStatus;
    private final Instant createdAt;
    private final Instant updatedAt;
    private final Instant completedAt;

    private TaskResult(
            String id,
            String taskId,
            int attemptNo,
            ResultType resultType,
            Map<String, Object> summary,
            Instant completedAt
    ) {
        this.id = requireText(id, "结果 ID 不能为空");
        this.taskId = requireText(taskId, "任务 ID 不能为空");
        this.attemptNo = requirePositive(attemptNo, "attemptNo 必须大于 0");
        this.resultType = Objects.requireNonNull(resultType, "结果类型不能为空");
        this.summary = validateSummary(summary);
        this.resultStatus = ResultStatus.COMPLETED;
        this.createdAt = Objects.requireNonNull(completedAt, "完成时间不能为空");
        this.updatedAt = completedAt;
        this.completedAt = completedAt;
    }

    public static TaskResult createCompleted(
            String id,
            String taskId,
            int attemptNo,
            ResultType resultType,
            Map<String, Object> summary,
            Instant completedAt
    ) {
        return new TaskResult(id, taskId, attemptNo, resultType, summary, completedAt);
    }

    public TaskResultUniqueKey uniqueKey() {
        return new TaskResultUniqueKey(taskId, attemptNo);
    }

    public String getId() {
        return id;
    }

    public String getTaskId() {
        return taskId;
    }

    public int getAttemptNo() {
        return attemptNo;
    }

    public ResultType getResultType() {
        return resultType;
    }

    public Map<String, Object> getSummary() {
        return summary;
    }

    public ResultStatus getResultStatus() {
        return resultStatus;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public Instant getCompletedAt() {
        return completedAt;
    }

    private static Map<String, Object> validateSummary(Map<String, Object> summary) {
        if (summary == null || summary.isEmpty() || !(summary.get("schema_version") instanceof Integer)) {
            throw new ResultBindingException("summary 必须包含 integer 类型 schema_version");
        }
        return Map.copyOf(new LinkedHashMap<>(summary));
    }

    private static String requireText(String value, String message) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    private static int requirePositive(int value, String message) {
        if (value <= 0) {
            throw new IllegalArgumentException(message);
        }
        return value;
    }

    public record TaskResultUniqueKey(String taskId, int attemptNo) {
    }
}
