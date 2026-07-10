package com.aiinterview.platform.application.result;

/**
 * 结果管理应用异常。
 */
public class ResultException extends RuntimeException {

    private final int code;

    public ResultException(int code, String message) {
        super(message);
        this.code = code;
    }

    public ResultException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
