package com.aiinterview.platform.domain.asyncmessage;

import java.util.Arrays;
import java.util.Optional;

/**
 * Worker 反馈类型。
 */
public enum WorkerFeedbackType {
    CONSUMED("consumed"),
    PROGRESS("progress"),
    PROBLEM_SET("problem_set"),
    MOCK_INTERVIEW_INTERMEDIATE("mock_interview_intermediate"),
    RESULT("result"),
    FAILURE("failure");

    private final String code;

    WorkerFeedbackType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }

    public static Optional<WorkerFeedbackType> fromCode(String code) {
        return Arrays.stream(values())
                .filter(type -> type.code.equals(code))
                .findFirst();
    }
}
