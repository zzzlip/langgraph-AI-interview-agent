package com.aiinterview.platform.application.resumeevaluation;

import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationCreatedResult;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationTaskCreation;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationTaskPort;
import com.aiinterview.platform.application.task.TaskApplicationService;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.file.FileBindType;
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
 * 默认简历评估任务创建端口。
 */
public class DefaultResumeEvaluationTaskPort implements ResumeEvaluationTaskPort {

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

    public DefaultResumeEvaluationTaskPort(
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
    public ResumeEvaluationCreatedResult createResumeEvaluationTask(ResumeEvaluationTaskCreation creation) {
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
                TaskType.RESUME_EVALUATION,
                creation.inputSummary(),
                creation.traceId(),
                now
        );
        TaskAttempt attempt = TaskAttempt.createInitial(idSupplier.get(), task.getId(), now);
        TaskEvent event = new TaskEvent(
                idSupplier.get(),
                task.getId(),
                task.getCurrentAttempt(),
                TaskEventType.TASK_CREATED,
                null,
                TaskStatus.PENDING,
                null,
                null,
                "简历评估任务已创建",
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                creation.userId(),
                creation.traceId(),
                now,
                now
        );
        FileBinding binding = FileBinding.createInput(
                idSupplier.get(),
                creation.resumeFileId(),
                task.getId(),
                creation.userId(),
                task.getCurrentAttempt(),
                now
        );
        ResumeEvaluationCreatedResult result = new ResumeEvaluationCreatedResult(
                task.getId(),
                task.getTaskType().getCode(),
                task.getStatus().getCode(),
                task.getCreatedAt()
        );

        try {
            taskRepository.save(task);
            attemptRepository.save(attempt);
            eventRepository.save(event);
            fileBindingRepository.save(binding);
            outboxPort.enqueueTaskCommand(task, TaskApplicationService.CREATE_TASK_OPERATION, creation.workerPayload());
            idempotencyRepository.save(toIdempotencyRecord(creation, task.getId(), result, now));
            auditWritePort.writeAudit("RESUME_EVALUATION_CREATED", creation.userId(), task.getId(), Map.of(
                    "resume_file_id", creation.resumeFileId(),
                    "trace_id", creation.traceId()
            ));
        } catch (DuplicateTaskIdempotencyException ex) {
            TaskIdempotencyRecord reloaded = idempotencyRepository
                    .findByUserIdAndOperationTypeAndIdempotencyKey(
                            creation.userId(),
                            TaskApplicationService.CREATE_TASK_OPERATION,
                            creation.idempotencyKey()
                    )
                    .orElseThrow(() -> new ResumeEvaluationException(
                            TASK_COMMAND_WRITE_FAILED_CODE,
                            "任务幂等记录冲突后重读失败",
                            ex
                    ));
            return resultFromIdempotency(creation, reloaded);
        } catch (ResumeEvaluationException ex) {
            throw ex;
        } catch (RuntimeException ex) {
            throw new ResumeEvaluationException(TASK_COMMAND_WRITE_FAILED_CODE, "任务命令写入失败", ex);
        }

        return result;
    }

    private ResumeEvaluationCreatedResult resultFromIdempotency(ResumeEvaluationTaskCreation creation, TaskIdempotencyRecord record) {
        if (!record.getRequestHash().equals(creation.requestHash())) {
            throw new ResumeEvaluationException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        return parseCreatedSnapshot(record.getResponseSnapshot());
    }

    private TaskIdempotencyRecord toIdempotencyRecord(
            ResumeEvaluationTaskCreation creation,
            String taskId,
            ResumeEvaluationCreatedResult result,
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

    private String toCreatedSnapshot(ResumeEvaluationCreatedResult result) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("task_id", result.taskId());
        snapshot.put("task_type", result.taskType());
        snapshot.put("status", result.status());
        snapshot.put("created_at", result.createdAt().toString());
        try {
            return objectMapper.writeValueAsString(snapshot);
        } catch (JsonProcessingException ex) {
            throw new ResumeEvaluationException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照序列化失败", ex);
        }
    }

    private ResumeEvaluationCreatedResult parseCreatedSnapshot(String responseSnapshot) {
        try {
            JsonNode root = objectMapper.readTree(responseSnapshot);
            return new ResumeEvaluationCreatedResult(
                    requiredText(root, "task_id"),
                    requiredText(root, "task_type"),
                    requiredText(root, "status"),
                    Instant.parse(requiredText(root, "created_at"))
            );
        } catch (RuntimeException | JsonProcessingException ex) {
            throw new ResumeEvaluationException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照格式非法", ex);
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
