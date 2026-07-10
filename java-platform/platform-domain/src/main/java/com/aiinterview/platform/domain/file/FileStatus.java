package com.aiinterview.platform.domain.file;

import java.util.Arrays;
import java.util.Optional;

/**
 * 文件状态。
 */
public enum FileStatus {
    UPLOADING("uploading", false),
    AVAILABLE("available", true),
    FAILED("failed", true);

    private final String code;
    private final boolean finalStatus;

    FileStatus(String code, boolean finalStatus) {
        this.code = code;
        this.finalStatus = finalStatus;
    }

    public String getCode() {
        return code;
    }

    /**
     * 当前状态是否为终态。
     */
    public boolean isFinal() {
        return finalStatus;
    }

    /**
     * 按外部 code 查找文件状态；未知值保持拒绝语义。
     */
    public static Optional<FileStatus> fromCode(String code) {
        return Arrays.stream(values())
                .filter(status -> status.code.equals(code))
                .findFirst();
    }
}
