package com.aiinterview.platform.application.asyncmessage;

/**
 * 异步消息应用异常。
 */
public class AsyncMessageException extends RuntimeException {

    private final int code;

    public AsyncMessageException(int code, String message) {
        super(message);
        this.code = code;
    }

    public AsyncMessageException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
