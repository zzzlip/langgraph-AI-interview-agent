package com.aiinterview.platform.domain.task;

/**
 * 任务主状态。
 */
public enum TaskStatus {
    PENDING("pending", false),
    QUEUED("queued", false),
    RUNNING("running", false),
    WAITING_USER_INPUT("waiting_user_input", false),
    SUCCESS("success", true),
    FAILED("failed", true),
    CANCELLED("cancelled", true);

    private final String code;
    private final boolean finalStatus;

    TaskStatus(String code, boolean finalStatus) {
        this.code = code;
        this.finalStatus = finalStatus;
    }

    public String getCode() {
        return code;
    }

    /**
     * 当前状态是否为终态。
     */
    public boolean isFinal() {
        return finalStatus;
    }
}
