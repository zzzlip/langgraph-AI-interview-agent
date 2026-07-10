package com.aiinterview.platform.interfaceadapter.http.task.request;

/**
 * 普通用户重试任务请求。
 */
public class RetryTaskRequest {

    private String reason;

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }
}
