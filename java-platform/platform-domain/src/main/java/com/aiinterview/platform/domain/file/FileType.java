package com.aiinterview.platform.domain.file;

import java.util.Arrays;
import java.util.Optional;

/**
 * V1 文件类型。
 */
public enum FileType {
    PDF("pdf", "application/pdf"),
    DOC("doc", "application/msword"),
    DOCX("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    PNG("png", "image/png"),
    JPG("jpg", "image/jpeg"),
    JPEG("jpeg", "image/jpeg"),
    MP3("mp3", "audio/mpeg"),
    MP4("mp4", "video/mp4"),
    WEBM("webm", "video/webm"),
    JSON("json", "application/json");

    private final String code;
    private final String mimeType;

    FileType(String code, String mimeType) {
        this.code = code;
        this.mimeType = mimeType;
    }

    public String getCode() {
        return code;
    }

    public String getMimeType() {
        return mimeType;
    }

    /**
     * 按外部 code 查找文件类型；未知值保持拒绝语义。
     */
    public static Optional<FileType> fromCode(String code) {
        return Arrays.stream(values())
                .filter(type -> type.code.equals(code))
                .findFirst();
    }
}
