package com.aiinterview.platform.domain.result;

import java.util.Arrays;
import java.util.Optional;

/**
 * 结果状态。
 */
public enum ResultStatus {
    COMPLETED("completed");

    private final String code;

    ResultStatus(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }

    /**
     * V1 只允许成功结果落库，其他状态保持拒绝语义。
     */
    public static Optional<ResultStatus> fromCode(String code) {
        return Arrays.stream(values())
                .filter(status -> status.code.equals(code))
                .findFirst();
    }
}
