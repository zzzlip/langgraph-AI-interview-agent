package com.aiinterview.platform.application.asyncmessage.port;

/**
 * 任务状态变化 SSE 发布端口。
 */
public interface SseTaskEventPublisher {

    void publishTaskChanged(String taskId);
}
