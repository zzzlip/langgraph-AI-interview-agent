package com.aiinterview.platform.application.mockinterview;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCreatedResult;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewTaskCreation;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewTaskPort;
import com.aiinterview.platform.application.task.TaskApplicationService;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.file.FileBinding;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.task.DuplicateTaskIdempotencyException;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskAttempt;
import com.aiinterview.platform.domain.task.TaskAttemptRepository;
import com.aiinterview.platform.domain.task.TaskEvent;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskEventType;
import com.aiinterview.platform.domain.task.TaskIdempotencyRecord;
import com.aiinterview.platform.domain.task.TaskIdempotencyRepository;
import com.aiinterview.platform.domain.task.TaskOperatorType;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.aiinterview.platform.domain.task.TaskType;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;

/**
 * 默认模拟面试任务端口。
 */
public class DefaultMockInterviewTaskPort implements MockInterviewTaskPort {

    private static final int IDEMPOTENCY_HASH_CONFLICT_CODE = 71008;
    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;

    private final TaskRepository taskRepository;
    private final TaskAttemptRepository attemptRepository;
    private final TaskEventRepository eventRepository;
    private final TaskIdempotencyRepository idempotencyRepository;
    private final FileBindingRepository fileBindingRepository;
    private final TaskCommandOutboxPort outboxPort;
    private final AuditWritePort auditWritePort;
    private final Clock clock;
    private final Supplier<String> idSupplier;
    private final ObjectMapper objectMapper;

    public DefaultMockInterviewTaskPort(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            FileBindingRepository fileBindingRepository,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier,
            ObjectMapper objectMapper
    ) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.attemptRepository = Objects.requireNonNull(attemptRepository, "attemptRepository 不能为空");
        this.eventRepository = Objects.requireNonNull(eventRepository, "eventRepository 不能为空");
        this.idempotencyRepository = Objects.requireNonNull(idempotencyRepository, "idempotencyRepository 不能为空");
        this.fileBindingRepository = Objects.requireNonNull(fileBindingRepository, "fileBindingRepository 不能为空");
        this.outboxPort = Objects.requireNonNull(outboxPort, "outboxPort 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    @Override
    @Transactional
    public MockInterviewCreatedResult createMockInterviewTask(MockInterviewTaskCreation creation) {
        TaskIdempotencyRecord existing = idempotencyRepository
                .findByUserIdAndOperationTypeAndIdempotencyKey(
                        creation.userId(),
                        TaskApplicationService.CREATE_TASK_OPERATION,
                        creation.idempotencyKey()
                )
                .orElse(null);
        if (existing != null) {
            return resultFromIdempotency(creation, existing);
        }

        Instant now = Instant.now(clock);
        Task task = Task.create(
                idSupplier.get(),
                creation.userId(),
                TaskType.MOCK_INTERVIEW,
                creation.inputSummary(),
                creation.traceId(),
                now
        );
        TaskAttempt attempt = TaskAttempt.createInitial(idSupplier.get(), task.getId(), now);
        FileBinding binding = FileBinding.createInput(
                idSupplier.get(),
                creation.resumeFileId(),
                task.getId(),
                creation.userId(),
                task.getCurrentAttempt(),
                now
        );
        TaskEvent event = new TaskEvent(
                idSupplier.get(),
                task.getId(),
                task.getCurrentAttempt(),
                TaskEventType.TASK_CREATED,
                null,
                TaskStatus.PENDING,
                null,
                null,
                "模拟面试任务已创建",
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                creation.userId(),
                creation.traceId(),
                now,
                now
        );
        MockInterviewCreatedResult result = new MockInterviewCreatedResult(
                task.getId(),
                task.getTaskType().getCode(),
                task.getStatus().getCode(),
                "排队中",
                task.getCreatedAt()
        );

        try {
            taskRepository.save(task);
            attemptRepository.save(attempt);
            fileBindingRepository.save(binding);
            eventRepository.save(event);
            outboxPort.enqueueTaskCommand(task, TaskApplicationService.CREATE_TASK_OPERATION, creation.workerPayload());
            idempotencyRepository.save(toIdempotencyRecord(creation, task.getId(), result, now));
            auditWritePort.writeAudit("MOCK_INTERVIEW_CREATED", creation.userId(), task.getId(), Map.of(
                    "trace_id", creation.traceId()
            ));
        } catch (DuplicateTaskIdempotencyException ex) {
            TaskIdempotencyRecord reloaded = idempotencyRepository
                    .findByUserIdAndOperationTypeAndIdempotencyKey(
                            creation.userId(),
                            TaskApplicationService.CREATE_TASK_OPERATION,
                            creation.idempotencyKey()
                    )
                    .orElseThrow(() -> new MockInterviewException(
                            TASK_COMMAND_WRITE_FAILED_CODE,
                            "任务幂等记录冲突后重读失败",
                            ex
                    ));
            return resultFromIdempotency(creation, reloaded);
        } catch (MockInterviewException ex) {
            throw ex;
        } catch (RuntimeException ex) {
            throw new MockInterviewException(TASK_COMMAND_WRITE_FAILED_CODE, "任务命令写入失败", ex);
        }

        return result;
    }

    private MockInterviewCreatedResult resultFromIdempotency(
            MockInterviewTaskCreation creation,
            TaskIdempotencyRecord record
    ) {
        if (!record.getRequestHash().equals(creation.requestHash())) {
            throw new MockInterviewException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        return parseCreatedSnapshot(record.getResponseSnapshot());
    }

    private TaskIdempotencyRecord toIdempotencyRecord(
            MockInterviewTaskCreation creation,
            String taskId,
            MockInterviewCreatedResult result,
            Instant now
    ) {
        return new TaskIdempotencyRecord(
                idSupplier.get(),
                creation.userId(),
                creation.idempotencyKey(),
                TaskApplicationService.CREATE_TASK_OPERATION,
                creation.requestHash(),
                taskId,
                toCreatedSnapshot(result),
                now.plusSeconds(86400),
                now,
                now
        );
    }

    private String toCreatedSnapshot(MockInterviewCreatedResult result) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("task_id", result.taskId());
        snapshot.put("task_type", result.taskType());
        snapshot.put("status", result.status());
        snapshot.put("status_label", result.statusLabel());
        snapshot.put("created_at", result.createdAt().toString());
        try {
            return objectMapper.writeValueAsString(snapshot);
        } catch (JsonProcessingException ex) {
            throw new MockInterviewException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照序列化失败", ex);
        }
    }

    private MockInterviewCreatedResult parseCreatedSnapshot(String responseSnapshot) {
        try {
            JsonNode root = objectMapper.readTree(responseSnapshot);
            return new MockInterviewCreatedResult(
                    requiredText(root, "task_id"),
                    requiredText(root, "task_type"),
                    requiredText(root, "status"),
                    requiredText(root, "status_label"),
                    Instant.parse(requiredText(root, "created_at"))
            );
        } catch (RuntimeException | JsonProcessingException ex) {
            throw new MockInterviewException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照格式非法", ex);
        }
    }

    private static String requiredText(JsonNode root, String field) {
        JsonNode value = root.get(field);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("缺少字段 " + field);
        }
        return value.asText();
    }
}
