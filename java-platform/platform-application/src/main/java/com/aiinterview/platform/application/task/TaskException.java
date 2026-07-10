package com.aiinterview.platform.application.task;

/**
 * 任务中心应用异常。
 */
public class TaskException extends RuntimeException {

    private final int code;

    public TaskException(int code, String message) {
        super(message);
        this.code = code;
    }

    public TaskException(int code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
