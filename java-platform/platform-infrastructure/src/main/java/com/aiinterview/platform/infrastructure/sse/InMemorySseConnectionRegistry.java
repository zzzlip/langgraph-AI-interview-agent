package com.aiinterview.platform.infrastructure.sse;

import com.aiinterview.platform.application.asyncmessage.dto.SseTaskEventResult;
import com.aiinterview.platform.application.asyncmessage.port.SseConnectionRegistry;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.springframework.stereotype.Component;

/**
 * 单机内存 SSE 连接注册表。
 */
@Component
public class InMemorySseConnectionRegistry implements SseConnectionRegistry {

    private final List<Connection> connections = new CopyOnWriteArrayList<>();

    @Override
    public void register(String userId, String taskId, TaskEventSink sink) {
        connections.add(new Connection(userId, taskId, sink));
    }

    @Override
    public void publish(String taskId, SseTaskEventResult event) {
        for (Connection connection : connections) {
            if (!connection.taskId().equals(taskId)) {
                continue;
            }
            try {
                connection.sink().send(event);
            } catch (RuntimeException ex) {
                connections.remove(connection);
            }
        }
    }

    @Override
    public int countByUser(String userId) {
        return (int) connections.stream().filter(connection -> connection.userId().equals(userId)).count();
    }

    @Override
    public int countByTask(String taskId) {
        return (int) connections.stream().filter(connection -> connection.taskId().equals(taskId)).count();
    }

    @Override
    public void remove(String userId, String taskId, TaskEventSink sink) {
        connections.removeIf(connection -> connection.userId().equals(userId)
                && connection.taskId().equals(taskId)
                && connection.sink() == sink);
    }

    private record Connection(String userId, String taskId, TaskEventSink sink) {
    }
}
