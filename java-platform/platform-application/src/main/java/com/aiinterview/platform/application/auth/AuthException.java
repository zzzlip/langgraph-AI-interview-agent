package com.aiinterview.platform.application.auth;

/**
 * 认证业务异常。
 */
public class AuthException extends RuntimeException {

    private final int code;

    public AuthException(int code, String message) {
        super(message);
        this.code = code;
    }

    public AuthException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
