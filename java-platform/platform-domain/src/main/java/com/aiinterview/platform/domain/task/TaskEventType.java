package com.aiinterview.platform.domain.task;

/**
 * 任务历史事件类型。
 */
public enum TaskEventType {
    TASK_CREATED("task_created"),
    TASK_QUEUED("task_queued"),
    TASK_RUNNING("task_running"),
    TASK_PROGRESS_UPDATED("task_progress_updated"),
    TASK_WAITING_USER_INPUT("task_waiting_user_input"),
    TASK_SUCCEEDED("task_succeeded"),
    TASK_FAILED("task_failed"),
    TASK_CANCELLED("task_cancelled"),
    TASK_RETRY_REQUESTED("task_retry_requested"),
    TASK_FEEDBACK_IGNORED("task_feedback_ignored");

    private final String code;

    TaskEventType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }
}
