package com.aiinterview.platform.interfaceadapter.http.file.response;

import com.aiinterview.platform.application.file.dto.FileDetailResult;
import com.aiinterview.platform.application.file.dto.FileUploadResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 文件元数据响应。
 */
public record FileResponse(
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
        @JsonProperty("file_status")
        String fileStatus,
        @JsonProperty("created_at")
        Instant createdAt
) {

    public static FileResponse from(FileUploadResult result) {
        return new FileResponse(
                result.fileId(),
                result.fileName(),
                result.fileType(),
                result.filePurpose(),
                result.fileSize(),
                result.fileStatus(),
                result.createdAt()
        );
    }

    public static FileResponse from(FileDetailResult result) {
        return new FileResponse(
                result.fileId(),
                result.fileName(),
                result.fileType(),
                result.filePurpose(),
                result.fileSize(),
                result.fileStatus(),
                result.createdAt()
        );
    }
}
