package com.aiinterview.platform.application.mockinterview;

/**
 * 模拟面试业务异常。
 */
public class MockInterviewException extends RuntimeException {

    private final int code;

    public MockInterviewException(int code, String message) {
        super(message);
        this.code = code;
    }

    public MockInterviewException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
