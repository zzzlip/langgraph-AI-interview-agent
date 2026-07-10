package com.aiinterview.platform.application.resumeoptimization;

/**
 * 简历优化业务异常。
 */
public class ResumeOptimizationException extends RuntimeException {

    private final int code;

    public ResumeOptimizationException(int code, String message) {
        super(message);
        this.code = code;
    }

    public ResumeOptimizationException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
