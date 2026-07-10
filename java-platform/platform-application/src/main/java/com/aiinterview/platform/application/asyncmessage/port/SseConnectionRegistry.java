package com.aiinterview.platform.application.asyncmessage.port;

import com.aiinterview.platform.application.asyncmessage.dto.SseTaskEventResult;

/**
 * SSE 连接注册表端口。
 */
public interface SseConnectionRegistry {

    void register(String userId, String taskId, TaskEventSink sink);

    void publish(String taskId, SseTaskEventResult event);

    int countByUser(String userId);

    int countByTask(String taskId);

    void remove(String userId, String taskId, TaskEventSink sink);

    interface TaskEventSink {
        void send(SseTaskEventResult event);
    }
}
