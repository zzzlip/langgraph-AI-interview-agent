package com.aiinterview.platform.application.file.command;

import java.util.Arrays;
import java.util.Objects;

/**
 * 文件上传命令。
 */
public record UploadFileCommand(
        String userId,
        String originalName,
        String filePurpose,
        String declaredMimeType,
        byte[] content,
        String sourceDeviceId,
        String traceId
) {

    public UploadFileCommand {
        Objects.requireNonNull(userId, "userId 不能为空");
        Objects.requireNonNull(originalName, "originalName 不能为空");
        Objects.requireNonNull(filePurpose, "filePurpose 不能为空");
        Objects.requireNonNull(declaredMimeType, "declaredMimeType 不能为空");
        Objects.requireNonNull(content, "content 不能为空");
        Objects.requireNonNull(traceId, "traceId 不能为空");
        content = Arrays.copyOf(content, content.length);
    }
}
