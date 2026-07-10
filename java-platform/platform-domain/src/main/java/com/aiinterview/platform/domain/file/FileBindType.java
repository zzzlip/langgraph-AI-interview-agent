package com.aiinterview.platform.domain.file;

/**
 * 文件与任务的绑定类型。
 */
public enum FileBindType {
    INPUT("input"),
    OUTPUT("output"),
    REPORT("report");

    private final String code;

    FileBindType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }
}
