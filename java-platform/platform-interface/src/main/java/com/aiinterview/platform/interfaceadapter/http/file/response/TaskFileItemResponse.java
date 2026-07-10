package com.aiinterview.platform.interfaceadapter.http.file.response;

import com.aiinterview.platform.application.file.dto.FileTaskFileItem;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 任务关联文件响应条目。
 */
public record TaskFileItemResponse(
        @JsonProperty("file_id")
        String fileId,
        @JsonProperty("file_name")
        String fileName,
        @JsonProperty("file_type")
        String fileType,
        @JsonProperty("file_purpose")
        String filePurpose,
        @JsonProperty("file_size")
        long fileSize,
        @JsonProperty("bind_type")
        String bindType,
        @JsonProperty("created_at")
        Instant createdAt
) {

    public static TaskFileItemResponse from(FileTaskFileItem item) {
        return new TaskFileItemResponse(
                item.fileId(),
                item.fileName(),
                item.fileType(),
                item.filePurpose(),
                item.fileSize(),
                item.bindType(),
                item.createdAt()
        );
    }
}
