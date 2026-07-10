package com.aiinterview.platform.domain.audit;

/**
 * 审计事件类型枚举。
 */
public enum AuditEventType {
    REGISTER_SUCCESS,
    REGISTER_FAILURE,
    LOGIN_SUCCESS,
    LOGIN_FAILURE,
    LOGOUT,
    PERMISSION_DENIED,
    FILE_DOWNLOAD,
    ADMIN_ACCOUNT_BOOTSTRAP,
    ADMIN_VIEW_FAILURE_DETAIL,
    ADMIN_RETRY_TASK,
    TASK_CANCEL,
    TASK_DELETE
}
