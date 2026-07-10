package com.aiinterview.platform.application.task.port;

import java.util.List;
import java.util.Objects;

/**
 * 任务成功结果引用绑定命令。
 */
public record BindResultReferenceCommand(
        String taskId,
        String userId,
        int attemptNo,
        String resultStatus,
        String resultType,
        String summary,
        List<OutputFilePayload> outputFiles,
        String traceId
) {

    public BindResultReferenceCommand {
        Objects.requireNonNull(taskId, "taskId 不能为空");
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(resultStatus, "resultStatus 不能为空");
        Objects.requireNonNull(resultType, "resultType 不能为空");
        Objects.requireNonNull(summary, "summary 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
        outputFiles = outputFiles == null ? List.of() : List.copyOf(outputFiles);
        if (attemptNo <= 0) {
            throw new IllegalArgumentException("attemptNo 必须大于 0");
        }
    }

    /**
     * Worker 输出文件元信息。
     */
    public record OutputFilePayload(
            String objectKey,
            String fileType,
            String filePurpose,
            String mimeType,
            long fileSize,
            String displayName
    ) {

        public OutputFilePayload {
            Objects.requireNonNull(objectKey, "objectKey 不能为空");
            Objects.requireNonNull(fileType, "fileType 不能为空");
            Objects.requireNonNull(filePurpose, "filePurpose 不能为空");
            Objects.requireNonNull(mimeType, "mimeType 不能为空");
            Objects.requireNonNull(displayName, "displayName 不能为空");
            if (fileSize <= 0) {
                throw new IllegalArgumentException("fileSize 必须大于 0");
            }
        }
    }
}
