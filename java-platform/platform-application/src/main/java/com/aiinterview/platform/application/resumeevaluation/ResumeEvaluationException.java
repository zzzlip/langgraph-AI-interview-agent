package com.aiinterview.platform.application.resumeevaluation;

/**
 * 简历评估业务异常。
 */
public class ResumeEvaluationException extends RuntimeException {

    private final int code;

    public ResumeEvaluationException(int code, String message) {
        super(message);
        this.code = code;
    }

    public ResumeEvaluationException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
