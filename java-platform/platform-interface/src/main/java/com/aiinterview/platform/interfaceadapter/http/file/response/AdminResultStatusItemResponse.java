package com.aiinterview.platform.interfaceadapter.http.file.response;

import com.aiinterview.platform.application.file.dto.AdminResultStatusItem;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 管理员结果状态响应项。
 */
public record AdminResultStatusItemResponse(
        @JsonProperty("result_id") String resultId,
        @JsonProperty("task_id") String taskId,
        @JsonProperty("attempt_no") int attemptNo,
        @JsonProperty("result_type") String resultType,
        @JsonProperty("result_status") String resultStatus,
        @JsonProperty("report_file_count") long reportFileCount,
        @JsonProperty("output_file_count") long outputFileCount,
        @JsonProperty("created_at") Instant createdAt,
        @JsonProperty("completed_at") Instant completedAt
) {
    public static AdminResultStatusItemResponse from(AdminResultStatusItem item) {
        return new AdminResultStatusItemResponse(
                item.resultId(),
                item.taskId(),
                item.attemptNo(),
                item.resultType(),
                item.resultStatus(),
                item.reportFileCount(),
                item.outputFileCount(),
                item.createdAt(),
                item.completedAt()
        );
    }
}
