package com.aiinterview.platform.application.file;

/**
 * 文件管理应用异常。
 */
public class FileException extends RuntimeException {

    private final int code;

    public FileException(int code, String message) {
        super(message);
        this.code = code;
    }

    public FileException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
