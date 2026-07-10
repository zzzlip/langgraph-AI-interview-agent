package com.aiinterview.platform.interfaceadapter.error;

/**
 * 资源不存在异常。
 * 对应 HTTP 404 + 错误码 30001。
 */
public class ResourceNotFoundException extends BusinessException {

    private static final int DEFAULT_CODE = 30001;
    private static final String DEFAULT_MESSAGE = "资源不存在";

    public ResourceNotFoundException() {
        super(DEFAULT_CODE, DEFAULT_MESSAGE);
    }

    public ResourceNotFoundException(String message) {
        super(DEFAULT_CODE, message);
    }

    public ResourceNotFoundException(String message, Throwable cause) {
        super(DEFAULT_CODE, message);
        initCause(cause);
    }
}
