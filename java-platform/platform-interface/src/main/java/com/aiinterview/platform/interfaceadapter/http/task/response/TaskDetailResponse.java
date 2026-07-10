package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskDetailResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * 任务详情响应。
 */
public record TaskDetailResponse(
        @JsonProperty("task_id") String taskId,
        @JsonProperty("task_type") String taskType,
        String status,
        @JsonProperty("status_label") String statusLabel,
        String stage,
        Integer progress,
        int attempt,
        @JsonProperty("retry_count") int retryCount,
        @JsonProperty("max_retry_count") int maxRetryCount,
        boolean retryable,
        boolean cancellable,
        @JsonProperty("input_summary") String inputSummary,
        @JsonProperty("result_reference") String resultReference,
        ResultDetailResponse result,
        TaskFailureResponse failure,
        List<TaskAttemptResponse> attempts,
        List<TaskEventResponse> events,
        @JsonProperty("created_at") Instant createdAt,
        @JsonProperty("updated_at") Instant updatedAt
) {
    public static TaskDetailResponse from(TaskDetailResult result) {
        return new TaskDetailResponse(
                result.taskId(),
                result.taskType(),
                result.status(),
                result.statusLabel(),
                result.stage(),
                result.progress(),
                result.attempt(),
                result.retryCount(),
                result.maxRetryCount(),
                result.retryable(),
                result.cancellable(),
                result.inputSummary(),
                result.resultReferenceId(),
                ResultDetailResponse.from(result.result()),
                TaskFailureResponse.from(result.failure()),
                result.attempts().stream().map(TaskAttemptResponse::from).toList(),
                result.events().stream().map(TaskEventResponse::from).toList(),
                result.createdAt(),
                result.updatedAt()
        );
    }

    public record ResultDetailResponse(
            @JsonProperty("result_id") String resultId,
            @JsonProperty("result_type") String resultType,
            Map<String, Object> summary,
            @JsonProperty("report_files") List<ResultFileResponse> reportFiles,
            @JsonProperty("output_files") List<ResultFileResponse> outputFiles,
            @JsonProperty("result_status") String resultStatus,
            @JsonProperty("created_at") Instant createdAt,
            @JsonProperty("completed_at") Instant completedAt
    ) {
        static ResultDetailResponse from(com.aiinterview.platform.application.result.dto.ResultDetailResult result) {
            if (result == null) {
                return null;
            }
            return new ResultDetailResponse(
                    result.resultId(),
                    result.resultType(),
                    result.summary(),
                    result.reportFiles().stream().map(ResultFileResponse::from).toList(),
                    result.outputFiles().stream().map(ResultFileResponse::from).toList(),
                    result.resultStatus(),
                    result.createdAt(),
                    result.completedAt()
            );
        }
    }

    public record ResultFileResponse(
            @JsonProperty("file_id") String fileId,
            @JsonProperty("file_name") String fileName,
            @JsonProperty("file_type") String fileType,
            @JsonProperty("file_size") long fileSize
    ) {
        static ResultFileResponse from(com.aiinterview.platform.application.result.dto.ResultFileItem item) {
            return new ResultFileResponse(item.fileId(), item.fileName(), item.fileType(), item.fileSize());
        }
    }
}
