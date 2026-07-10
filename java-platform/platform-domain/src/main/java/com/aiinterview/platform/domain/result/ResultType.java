package com.aiinterview.platform.domain.result;

import java.util.Arrays;
import java.util.Optional;

/**
 * V1 结果类型。
 */
public enum ResultType {
    RESUME_EVALUATION("resume_evaluation"),
    RESUME_OPTIMIZATION("resume_optimization"),
    ALGORITHM_TEST("algorithm_test"),
    MOCK_INTERVIEW("mock_interview");

    private final String code;

    ResultType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }

    /**
     * 按外部 code 查找结果类型；未知值保持拒绝语义。
     */
    public static Optional<ResultType> fromCode(String code) {
        return Arrays.stream(values())
                .filter(type -> type.code.equals(code))
                .findFirst();
    }
}
