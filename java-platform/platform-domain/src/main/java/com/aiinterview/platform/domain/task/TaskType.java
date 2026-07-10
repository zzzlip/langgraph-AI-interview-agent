package com.aiinterview.platform.domain.task;

/**
 * V1 任务类型。
 */
public enum TaskType {
    RESUME_EVALUATION("resume_evaluation"),
    RESUME_OPTIMIZATION("resume_optimization"),
    ALGORITHM_TEST("algorithm_test"),
    MOCK_INTERVIEW("mock_interview");

    private final String code;

    TaskType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }
}
