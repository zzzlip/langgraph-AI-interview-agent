package com.aiinterview.platform.application.security;

/**
 * 安全相关异常。
 */
public class SecurityException extends RuntimeException {

    private final int code;

    public SecurityException(int code, String message) {
        super(message);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
