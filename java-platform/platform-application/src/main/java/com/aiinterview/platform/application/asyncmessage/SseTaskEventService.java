package com.aiinterview.platform.application.asyncmessage;

import com.aiinterview.platform.application.asyncmessage.dto.SseTaskEventResult;
import com.aiinterview.platform.application.asyncmessage.port.SseConnectionRegistry;
import com.aiinterview.platform.application.asyncmessage.port.SseTaskEventPublisher;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskEvent;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.domain.task.TaskStatus;
import java.time.Clock;
import java.time.Instant;
import java.util.Comparator;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * 任务 SSE 订阅与推送应用服务。
 */
public class SseTaskEventService implements SseTaskEventPublisher {

    private static final int NOT_FOUND_MASK_CODE = 30001;
    private static final int CONNECTION_UNAVAILABLE_CODE = 74013;
    private static final int LIMIT_EXCEEDED_CODE = 74014;

    private final TaskRepository taskRepository;
    private final TaskEventRepository taskEventRepository;
    private final SseConnectionRegistry registry;
    private final AuditWritePort auditWritePort;
    private final Clock clock;
    private final int maxConnectionsPerUser;
    private final int maxConnectionsPerTask;

    public SseTaskEventService(TaskRepository taskRepository, TaskEventRepository taskEventRepository,
            SseConnectionRegistry registry,
            AuditWritePort auditWritePort, Clock clock, int maxConnectionsPerUser, int maxConnectionsPerTask) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.taskEventRepository = Objects.requireNonNull(taskEventRepository, "taskEventRepository 不能为空");
        this.registry = Objects.requireNonNull(registry, "registry 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.maxConnectionsPerUser = maxConnectionsPerUser;
        this.maxConnectionsPerTask = maxConnectionsPerTask;
    }

    public SseTaskEventResult subscribe(String userId, String taskId, SseConnectionRegistry.TaskEventSink sink) {
        Task task = taskRepository.findById(taskId)
                .filter(candidate -> candidate.getUserId().equals(userId))
                .orElseThrow(() -> {
                    auditWritePort.writeAudit("SSE_SUBSCRIBE_DENIED", userId, taskId, Map.of("reason", "not_found"));
                    return new com.aiinterview.platform.application.security.SecurityException(NOT_FOUND_MASK_CODE, "资源不存在");
                });
        if (registry.countByUser(userId) >= maxConnectionsPerUser
                || registry.countByTask(taskId) >= maxConnectionsPerTask) {
            throw new AsyncMessageException(LIMIT_EXCEEDED_CODE, "SSE 连接数超限");
        }
        registry.register(userId, taskId, sink);
        SseTaskEventResult snapshot = toEvent("task_snapshot", task, Instant.now(clock));
        try {
            sink.send(snapshot);
        } catch (RuntimeException ex) {
            registry.remove(userId, taskId, sink);
            throw new AsyncMessageException(CONNECTION_UNAVAILABLE_CODE, "SSE 连接不可用", ex);
        }
        return snapshot;
    }

    public void disconnect(String userId, String taskId, SseConnectionRegistry.TaskEventSink sink) {
        registry.remove(userId, taskId, sink);
    }

    @Override
    public void publishTaskChanged(String taskId) {
        taskRepository.findById(taskId)
                .map(task -> toEvent(eventName(task), task, Instant.now(clock)))
                .ifPresent(event -> registry.publish(taskId, event));
    }

    private static String eventName(Task task) {
        if (task.getStatus() == TaskStatus.SUCCESS) {
            return "task_completed";
        }
        if (task.getStatus() == TaskStatus.FAILED) {
            return "task_failed";
        }
        return "task_progress";
    }

    private SseTaskEventResult toEvent(String eventName, Task task, Instant now) {
        return new SseTaskEventResult(
                eventName,
                task.getId() + ":" + task.getCurrentAttempt() + ":" + now.toEpochMilli(),
                task.getId(),
                task.getStatus().getCode(),
                task.getStage(),
                task.getProgress(),
                task.getCurrentAttempt(),
                latestVisibleMessage(task).orElse(null),
                task.isRetryable(),
                task.getTraceId(),
                now
        );
    }

    private Optional<String> latestVisibleMessage(Task task) {
        return taskEventRepository.findByTaskIdOrderByOccurredAt(task.getId()).stream()
                .filter(event -> event.getAttemptNo() == null || event.getAttemptNo() == task.getCurrentAttempt())
                .filter(event -> event.getMessage() != null && !event.getMessage().isBlank())
                .max(Comparator.comparing(TaskEvent::getOccurredAt))
                .map(TaskEvent::getMessage);
    }
}
