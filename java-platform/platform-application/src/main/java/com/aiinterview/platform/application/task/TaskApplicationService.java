package com.aiinterview.platform.application.task;

import com.aiinterview.platform.application.task.command.CreateTaskCommand;
import com.aiinterview.platform.application.task.command.CancelTaskCommand;
import com.aiinterview.platform.application.task.command.RetryTaskCommand;
import com.aiinterview.platform.application.task.dto.TaskCreatedResult;
import com.aiinterview.platform.application.task.dto.TaskStatusResult;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.ResourceOwnershipPort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
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
import java.time.Clock;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;

/**
 * 任务应用服务。
 */
public class TaskApplicationService {

    public static final String CREATE_TASK_OPERATION = "create_task";
    public static final String USER_RETRY_TASK_OPERATION = "user_retry_task";

    private static final int IDEMPOTENCY_HASH_CONFLICT_CODE = 71008;
    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;
    private static final int RESOURCE_OWNERSHIP_DENIED_CODE = 71003;
    private static final int TASK_NOT_FOUND_CODE = 71003;
    private static final int TASK_STATUS_NOT_ALLOWED_CODE = 71004;
    private static final int TASK_NOT_CANCELLABLE_CODE = 71006;
    private static final int TASK_NOT_RETRYABLE_CODE = 71007;
    private static final int TASK_VERSION_CONFLICT_CODE = 71009;
    private static final int TASK_RETRY_LIMIT_REACHED_CODE = 71015;

    private final TaskRepository taskRepository;
    private final TaskAttemptRepository attemptRepository;
    private final TaskEventRepository eventRepository;
    private final TaskIdempotencyRepository idempotencyRepository;
    private final TaskCommandOutboxPort outboxPort;
    private final ResourceOwnershipPort resourceOwnershipPort;
    private final AuditWritePort auditWritePort;
    private final Clock clock;
    private final Supplier<String> idSupplier;

    public TaskApplicationService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            TaskCommandOutboxPort outboxPort,
            ResourceOwnershipPort resourceOwnershipPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.attemptRepository = Objects.requireNonNull(attemptRepository, "attemptRepository 不能为空");
        this.eventRepository = Objects.requireNonNull(eventRepository, "eventRepository 不能为空");
        this.idempotencyRepository = Objects.requireNonNull(idempotencyRepository, "idempotencyRepository 不能为空");
        this.outboxPort = Objects.requireNonNull(outboxPort, "outboxPort 不能为空");
        this.resourceOwnershipPort = Objects.requireNonNull(resourceOwnershipPort, "resourceOwnershipPort 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
    }

    /**
     * 创建任务，成功副作用必须和幂等记录处于同一事务边界。
     */
    @Transactional
    public TaskCreatedResult createTask(CreateTaskCommand command) {
        TaskIdempotencyRecord existing = idempotencyRepository
                .findByUserIdAndOperationTypeAndIdempotencyKey(
                        command.userId(),
                        CREATE_TASK_OPERATION,
                        command.idempotencyKey()
                )
                .orElse(null);
        if (existing != null) {
            return resultFromIdempotency(command, existing);
        }

        if (!resourceOwnershipPort.isOwnedByUser(command.userId(), command.resourceType(), command.resourceId())) {
            auditWritePort.writeAudit(
                    "TASK_ACCESS_DENIED",
                    command.userId(),
                    command.resourceId(),
                    Map.of("resource_type", command.resourceType())
            );
            throw new TaskException(RESOURCE_OWNERSHIP_DENIED_CODE, "任务不存在");
        }

        Instant now = Instant.now(clock);
        Task task = Task.create(idSupplier.get(), command.userId(), command.taskType(), command.inputSummary(), command.traceId(), now);
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
                "任务已创建",
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                command.userId(),
                command.traceId(),
                now,
                now
        );
        TaskCreatedResult result = new TaskCreatedResult(task.getId(), task.getStatus().getCode());

        try {
            taskRepository.save(task);
            attemptRepository.save(attempt);
            eventRepository.save(event);
            outboxPort.enqueueTaskCommand(task, CREATE_TASK_OPERATION);
            idempotencyRepository.save(toIdempotencyRecord(command, task.getId(), result, now));
        } catch (DuplicateTaskIdempotencyException ex) {
            TaskIdempotencyRecord reloaded = idempotencyRepository
                    .findByUserIdAndOperationTypeAndIdempotencyKey(
                            command.userId(),
                            CREATE_TASK_OPERATION,
                            command.idempotencyKey()
                    )
                    .orElseThrow(() -> new TaskException(TASK_COMMAND_WRITE_FAILED_CODE, "任务幂等记录冲突后重读失败", ex));
            return resultFromIdempotency(command, reloaded);
        } catch (TaskException ex) {
            throw ex;
        } catch (RuntimeException ex) {
            throw new TaskException(TASK_COMMAND_WRITE_FAILED_CODE, "任务命令写入失败", ex);
        }

        return result;
    }

    /**
     * 取消当前用户自己的未终态任务。
     */
    @Transactional
    public TaskStatusResult cancelTask(CancelTaskCommand command) {
        Task task = taskRepository.findById(command.taskId())
                .orElseThrow(() -> new TaskException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (!task.getUserId().equals(command.userId())) {
            auditWritePort.writeAudit("TASK_ACCESS_DENIED", command.userId(), command.taskId(), Map.of("operation", "task_cancel"));
            throw new TaskException(TASK_NOT_FOUND_CODE, "任务不存在");
        }
        if (!task.isCancellable()) {
            throw new TaskException(TASK_NOT_CANCELLABLE_CODE, "任务不可取消");
        }

        TaskStatus fromStatus = task.getStatus();
        long expectedVersion = task.getVersion();
        Instant now = Instant.now(clock);
        task.cancel(command.userId(), now);

        if (!taskRepository.updateByIdAndVersion(task, expectedVersion)) {
            throw new TaskException(TASK_VERSION_CONFLICT_CODE, "任务版本冲突");
        }

        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                task.getCurrentAttempt(),
                TaskEventType.TASK_CANCELLED,
                fromStatus,
                TaskStatus.CANCELLED,
                task.getStage(),
                task.getProgress(),
                "任务已取消",
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                command.userId(),
                command.traceId(),
                now,
                now
        ));
        auditWritePort.writeAudit("TASK_CANCELLED", command.userId(), task.getId(), Map.of("status", task.getStatus().getCode()));

        return new TaskStatusResult(
                task.getId(),
                task.getStatus().getCode(),
                task.getCancelledAt(),
                task.getCurrentAttempt(),
                task.getRetryCount(),
                task.getMaxRetryCount()
        );
    }

    /**
     * 普通用户重试自己的可恢复失败任务。
     */
    @Transactional
    public TaskStatusResult retryTask(RetryTaskCommand command) {
        TaskIdempotencyRecord existing = idempotencyRepository
                .findByUserIdAndOperationTypeAndIdempotencyKey(
                        command.userId(),
                        USER_RETRY_TASK_OPERATION,
                        command.idempotencyKey()
                )
                .orElse(null);
        if (existing != null) {
            return statusResultFromIdempotency(command.requestHash(), existing);
        }

        Task task = taskRepository.findById(command.taskId())
                .orElseThrow(() -> new TaskException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (!task.getUserId().equals(command.userId())) {
            auditWritePort.writeAudit("TASK_ACCESS_DENIED", command.userId(), command.taskId(), Map.of("operation", "task_retry"));
            throw new TaskException(TASK_NOT_FOUND_CODE, "任务不存在");
        }
        if (task.getStatus() != TaskStatus.FAILED) {
            throw new TaskException(TASK_STATUS_NOT_ALLOWED_CODE, "任务状态不允许当前操作");
        }
        if (!task.isRetryable()) {
            throw new TaskException(TASK_NOT_RETRYABLE_CODE, "任务不可重试");
        }
        if (task.getRetryCount() >= task.getMaxRetryCount()) {
            throw new TaskException(TASK_RETRY_LIMIT_REACHED_CODE, "重试次数已达上限");
        }

        TaskStatus fromStatus = task.getStatus();
        long expectedVersion = task.getVersion();
        Instant now = Instant.now(clock);
        task.retry(command.userId(), now);
        TaskStatusResult result = new TaskStatusResult(
                task.getId(),
                task.getStatus().getCode(),
                task.getCancelledAt(),
                task.getCurrentAttempt(),
                task.getRetryCount(),
                task.getMaxRetryCount()
        );

        if (!taskRepository.updateByIdAndVersion(task, expectedVersion)) {
            throw new TaskException(TASK_VERSION_CONFLICT_CODE, "任务版本冲突");
        }

        attemptRepository.save(new TaskAttempt(
                idSupplier.get(),
                task.getId(),
                task.getCurrentAttempt(),
                TaskStatus.QUEUED,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                now,
                now
        ));
        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                task.getCurrentAttempt(),
                TaskEventType.TASK_RETRY_REQUESTED,
                fromStatus,
                TaskStatus.QUEUED,
                task.getStage(),
                task.getProgress(),
                command.reason(),
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                command.userId(),
                command.traceId(),
                now,
                now
        ));
        outboxPort.enqueueTaskCommand(task, USER_RETRY_TASK_OPERATION);
        auditWritePort.writeAudit("TASK_RETRY_REQUESTED", command.userId(), task.getId(), Map.of("attempt", task.getCurrentAttempt()));
        idempotencyRepository.save(toRetryIdempotencyRecord(command, result, now));

        return result;
    }

    private TaskCreatedResult resultFromIdempotency(CreateTaskCommand command, TaskIdempotencyRecord record) {
        if (!record.getRequestHash().equals(command.requestHash())) {
            throw new TaskException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        return TaskIdempotencySnapshotCodec.parseCreatedSnapshot(record.getResponseSnapshot());
    }

    private TaskIdempotencyRecord toIdempotencyRecord(CreateTaskCommand command, String taskId, TaskCreatedResult result, Instant now) {
        return new TaskIdempotencyRecord(
                idSupplier.get(),
                command.userId(),
                command.idempotencyKey(),
                CREATE_TASK_OPERATION,
                command.requestHash(),
                taskId,
                TaskIdempotencySnapshotCodec.toCreatedSnapshot(result),
                now.plusSeconds(86400),
                now,
                now
        );
    }

    private TaskIdempotencyRecord toRetryIdempotencyRecord(RetryTaskCommand command, TaskStatusResult result, Instant now) {
        return new TaskIdempotencyRecord(
                idSupplier.get(),
                command.userId(),
                command.idempotencyKey(),
                USER_RETRY_TASK_OPERATION,
                command.requestHash(),
                command.taskId(),
                TaskIdempotencySnapshotCodec.toStatusSnapshot(result),
                now.plusSeconds(86400),
                now,
                now
        );
    }

    private static TaskStatusResult statusResultFromIdempotency(String requestHash, TaskIdempotencyRecord record) {
        if (!record.getRequestHash().equals(requestHash)) {
            throw new TaskException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        return TaskIdempotencySnapshotCodec.parseStatusSnapshot(record.getResponseSnapshot());
    }
}
