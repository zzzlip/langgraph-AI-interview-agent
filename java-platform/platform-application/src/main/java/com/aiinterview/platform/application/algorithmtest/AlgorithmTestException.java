package com.aiinterview.platform.application.algorithmtest;

/**
 * 算法测试业务异常。
 */
public class AlgorithmTestException extends RuntimeException {

    private final int code;

    public AlgorithmTestException(int code, String message) {
        super(message);
        this.code = code;
    }

    public AlgorithmTestException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
