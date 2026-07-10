package com.aiinterview.platform.application.audit;

/**
 * 审计写入失败异常。
 */
public class AuditWriteException extends RuntimeException {

    public AuditWriteException(String message, Throwable cause) {
        super(message, cause);
    }
}
