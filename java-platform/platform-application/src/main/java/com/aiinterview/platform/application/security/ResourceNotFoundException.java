package com.aiinterview.platform.application.security;

/**
 * 资源不存在异常。
 */
public class ResourceNotFoundException extends SecurityException {

    private static final int CODE = 30001;
    private static final String MESSAGE = "资源不存在";

    public ResourceNotFoundException() {
        super(CODE, MESSAGE);
    }
}
