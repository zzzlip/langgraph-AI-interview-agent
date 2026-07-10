package com.aiinterview.platform.application.asyncmessage;

import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.asyncmessage.AsyncDeliveryStatus;
import com.aiinterview.platform.domain.asyncmessage.OutboxEvent;
import com.aiinterview.platform.domain.asyncmessage.OutboxEventRepository;
import com.aiinterview.platform.domain.task.Task;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.time.Clock;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

/**
 * 任务命令 Outbox 写入服务。
 */
public class TaskCommandOutboxService implements TaskCommandOutboxPort {

    private static final int COMMAND_TYPE_UNSUPPORTED_CODE = 74001;
    private static final int OUTBOX_WRITE_FAILED_CODE = 74002;
    private static final int DEFAULT_MAX_RETRY_COUNT = 5;
    private static final Set<String> SUPPORTED_COMMAND_TYPES = Set.of(
            "create_task",
            "retry_task",
            "user_retry_task",
            "admin_retry_task"
    );

    private final OutboxEventRepository repository;
    private final Clock clock;
    private final Supplier<String> idSupplier;
    private final ObjectMapper objectMapper;

    public TaskCommandOutboxService(OutboxEventRepository repository, Clock clock, Supplier<String> idSupplier,
            ObjectMapper objectMapper) {
        this.repository = Objects.requireNonNull(repository, "repository 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    @Override
    public void enqueueTaskCommand(Task task, String commandType) {
        enqueueTaskCommand(task, commandType, Map.of());
    }

    @Override
    public void enqueueTaskCommand(Task task, String commandType, Map<String, Object> payloadExtensions) {
        Objects.requireNonNull(task, "task 不能为空");
        Objects.requireNonNull(payloadExtensions, "payloadExtensions 不能为空");
        if (!SUPPORTED_COMMAND_TYPES.contains(commandType)) {
            throw new AsyncMessageException(COMMAND_TYPE_UNSUPPORTED_CODE, "任务命令类型不支持");
        }

        String eventType = commandEventType(task);
        Instant now = Instant.now(clock);
        String eventId = idSupplier.get();
        OutboxEvent event = new OutboxEvent(
                idSupplier.get(),
                eventId,
                "task",
                task.getId(),
                task.getId(),
                task.getTaskType().getCode(),
                task.getCurrentAttempt(),
                eventType,
                routingKey(task),
                payload(task, eventId, eventType, now, payloadExtensions),
                AsyncDeliveryStatus.PENDING,
                0,
                DEFAULT_MAX_RETRY_COUNT,
                now,
                null,
                task.getTraceId(),
                now,
                now,
                null
        );

        try {
            repository.save(event);
        } catch (RuntimeException ex) {
            throw new AsyncMessageException(OUTBOX_WRITE_FAILED_CODE, "Outbox 写入失败", ex);
        }
    }

    private static String commandEventType(Task task) {
        return task.getTaskType().getCode() + ".command.created";
    }

    private static String routingKey(Task task) {
        return "task." + task.getTaskType().getCode() + ".command";
    }

    private String payload(Task task, String eventId, String eventType, Instant occurredAt, Map<String, Object> payloadExtensions) {
        ObjectNode root = objectMapper.createObjectNode();
        root.put("schema_version", 1);
        root.put("event_id", eventId);
        root.put("event_type", eventType);
        root.put("task_id", task.getId());
        root.put("task_type", task.getTaskType().getCode());
        root.put("attempt", task.getCurrentAttempt());
        root.put("trace_id", task.getTraceId());
        root.put("occurred_at", occurredAt.toString());

        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("user_id", task.getUserId());
        Object inputSummary = payloadExtensions.get("input_summary");
        payload.set("input_summary", inputSummary == null
                ? inputSummaryNode(task.getInputSummary())
                : objectMapper.valueToTree(inputSummary));
        payloadExtensions.forEach((key, value) -> {
            if (!"user_id".equals(key) && !"input_summary".equals(key)) {
                payload.set(key, objectMapper.valueToTree(value));
            }
        });
        root.set("payload", payload);

        try {
            return objectMapper.writeValueAsString(root);
        } catch (JsonProcessingException ex) {
            throw new AsyncMessageException(OUTBOX_WRITE_FAILED_CODE, "Outbox payload 序列化失败", ex);
        }
    }

    private JsonNode inputSummaryNode(String inputSummary) {
        try {
            return objectMapper.readTree(inputSummary);
        } catch (JsonProcessingException ex) {
            return objectMapper.getNodeFactory().textNode(inputSummary);
        }
    }
}
