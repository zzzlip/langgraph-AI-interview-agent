package com.aiinterview.platform.application.task.port;

import com.aiinterview.platform.domain.task.Task;
import java.util.Map;

/**
 * 任务命令 Outbox 写入端口。
 */
public interface TaskCommandOutboxPort {

    void enqueueTaskCommand(Task task, String commandType);

    default void enqueueTaskCommand(Task task, String commandType, Map<String, Object> payload) {
        enqueueTaskCommand(task, commandType);
    }
}
