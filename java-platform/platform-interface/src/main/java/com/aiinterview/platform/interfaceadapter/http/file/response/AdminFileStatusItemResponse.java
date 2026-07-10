package com.aiinterview.platform.interfaceadapter.http.file.response;

import com.aiinterview.platform.application.file.dto.AdminFileStatusItem;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 管理员文件状态响应项。
 */
public record AdminFileStatusItemResponse(
        @JsonProperty("file_id") String fileId,
        @JsonProperty("task_id") String taskId,
        @JsonProperty("file_type") String fileType,
        @JsonProperty("file_purpose") String filePurpose,
        @JsonProperty("file_status") String fileStatus,
        @JsonProperty("failure_reason") String failureReason,
        @JsonProperty("created_at") Instant createdAt,
        @JsonProperty("updated_at") Instant updatedAt
) {
    public static AdminFileStatusItemResponse from(AdminFileStatusItem item) {
        return new AdminFileStatusItemResponse(
                item.fileId(),
                item.taskId(),
                item.fileType(),
                item.filePurpose(),
                item.fileStatus(),
                item.failureReason(),
                item.createdAt(),
                item.updatedAt()
        );
    }
}
