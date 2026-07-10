package com.aiinterview.platform.domain.task;

/**
 * 任务事件操作者类型。
 */
public enum TaskOperatorType {
    USER("USER"),
    ADMIN("ADMIN"),
    WORKER("WORKER"),
    SYSTEM("SYSTEM");

    private final String code;

    TaskOperatorType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }
}
