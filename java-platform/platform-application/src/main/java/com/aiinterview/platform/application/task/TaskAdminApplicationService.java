package com.aiinterview.platform.application.task;

import com.aiinterview.platform.application.task.command.AdminRetryTaskCommand;
import com.aiinterview.platform.application.task.dto.AdminFailedTaskItem;
import com.aiinterview.platform.application.task.dto.AdminFailedTaskPageResult;
import com.aiinterview.platform.application.task.dto.TaskAttemptResult;
import com.aiinterview.platform.application.task.dto.TaskEventResult;
import com.aiinterview.platform.application.task.dto.TaskFailureDetailResult;
import com.aiinterview.platform.application.task.dto.TaskStatusResult;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.task.FailedTaskQueryCriteria;
import com.aiinterview.platform.domain.task.PageResult;
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
import java.time.Clock;
import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;

/**
 * 管理员任务排查应用服务。
 */
public class TaskAdminApplicationService {

    public static final String ADMIN_RETRY_TASK_OPERATION = "admin_retry_task";

    private static final int FORBIDDEN_CODE = 20003;
    private static final int TASK_NOT_FOUND_CODE = 71003;
    private static final int TASK_STATUS_NOT_ALLOWED_CODE = 71004;
    private static final int TASK_NOT_RETRYABLE_CODE = 71007;
    private static final int IDEMPOTENCY_HASH_CONFLICT_CODE = 71008;
    private static final int TASK_VERSION_CONFLICT_CODE = 71009;
    private static final int TASK_RETRY_LIMIT_REACHED_CODE = 71015;

    private final TaskRepository taskRepository;
    private final TaskAttemptRepository attemptRepository;
    private final TaskEventRepository eventRepository;
    private final TaskIdempotencyRepository idempotencyRepository;
    private final TaskCommandOutboxPort outboxPort;
    private final AuditWritePort auditWritePort;
    private final Clock clock;
    private final Supplier<String> idSupplier;

    public TaskAdminApplicationService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.attemptRepository = Objects.requireNonNull(attemptRepository, "attemptRepository 不能为空");
        this.eventRepository = Objects.requireNonNull(eventRepository, "eventRepository 不能为空");
        this.idempotencyRepository = Objects.requireNonNull(idempotencyRepository, "idempotencyRepository 不能为空");
        this.outboxPort = Objects.requireNonNull(outboxPort, "outboxPort 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
    }

    public AdminFailedTaskPageResult getFailedTaskList(String adminId, String role, TaskType taskType, Boolean retryable, int page, int pageSize) {
        requireAdmin(role);
        PageResult<Task> result = taskRepository.findFailed(new FailedTaskQueryCriteria(taskType, retryable, page, pageSize));
        List<AdminFailedTaskItem> items = result.items().stream()
                .map(this::toAdminFailedTaskItem)
                .toList();
        auditWritePort.writeAudit("TASK_ADMIN_FAILURE_VIEWED", adminId, "failed_tasks", Map.of("scope", "list"));
        return new AdminFailedTaskPageResult(items, result.total(), result.page(), result.pageSize());
    }

    public TaskFailureDetailResult getFailureDetail(String adminId, String role, String taskId) {
        requireAdmin(role);
        Task task = taskRepository.findById(taskId)
                .orElseThrow(() -> new TaskException(TASK_NOT_FOUND_CODE, "任务不存在"));
        List<TaskAttempt> attempts = attemptRepository.findByTaskId(taskId);
        List<TaskEvent> events = eventRepository.findByTaskIdOrderByOccurredAt(taskId);
        TaskAttempt failed = latestFailedAttempt(attempts);
        auditWritePort.writeAudit("TASK_ADMIN_FAILURE_VIEWED", adminId, taskId, Map.of("scope", "detail"));
        return new TaskFailureDetailResult(
                task.getId(),
                failed == null ? null : failed.getFailureStage(),
                failed == null ? null : failed.getFailureCode(),
                failed == null ? null : sanitize(failed.getAdminSummary()),
                attempts.stream().map(this::toAttemptResult).toList(),
                events.stream().map(this::toEventResult).toList(),
                task.getTraceId()
        );
    }

    @Transactional
    public TaskStatusResult adminRetryTask(AdminRetryTaskCommand command) {
        requireAdmin(command.role());
        TaskIdempotencyRecord existing = idempotencyRepository
                .findByUserIdAndOperationTypeAndIdempotencyKey(command.adminId(), ADMIN_RETRY_TASK_OPERATION, command.idempotencyKey())
                .orElse(null);
        if (existing != null) {
            return statusResultFromIdempotency(command.requestHash(), existing);
        }

        Task task = taskRepository.findById(command.taskId())
                .orElseThrow(() -> new TaskException(TASK_NOT_FOUND_CODE, "任务不存在"));
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
        task.retry(command.adminId(), now);
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
        attemptRepository.save(new TaskAttempt(idSupplier.get(), task.getId(), task.getCurrentAttempt(), TaskStatus.QUEUED,
                null, null, null, null, null, null, null, now, now));
        eventRepository.save(new TaskEvent(idSupplier.get(), task.getId(), task.getCurrentAttempt(), TaskEventType.TASK_RETRY_REQUESTED,
                fromStatus, TaskStatus.QUEUED, task.getStage(), task.getProgress(), command.reason(), Map.of("schema_version", 1),
                TaskOperatorType.ADMIN, command.adminId(), command.traceId(), now, now));
        outboxPort.enqueueTaskCommand(task, ADMIN_RETRY_TASK_OPERATION);
        auditWritePort.writeAudit("TASK_ADMIN_RETRY_REQUESTED", command.adminId(), task.getId(), Map.of("attempt", task.getCurrentAttempt()));
        idempotencyRepository.save(new TaskIdempotencyRecord(idSupplier.get(), command.adminId(), command.idempotencyKey(),
                ADMIN_RETRY_TASK_OPERATION, command.requestHash(), task.getId(), TaskIdempotencySnapshotCodec.toStatusSnapshot(result),
                now.plusSeconds(86400), now, now));
        return result;
    }

    private AdminFailedTaskItem toAdminFailedTaskItem(Task task) {
        TaskAttempt failed = latestFailedAttempt(attemptRepository.findByTaskId(task.getId()));
        return new AdminFailedTaskItem(
                task.getId(),
                task.getUserId(),
                task.getTaskType().getCode(),
                failed == null ? task.getStage() : failed.getFailureStage(),
                failed == null ? null : sanitize(failed.getAdminSummary()),
                task.getCurrentAttempt(),
                task.isRetryable(),
                task.getLastFailedAt(),
                task.getTraceId()
        );
    }

    private TaskAttempt latestFailedAttempt(List<TaskAttempt> attempts) {
        return attempts.stream()
                .filter(attempt -> attempt.getFailedAt() != null)
                .max(Comparator.comparing(TaskAttempt::getFailedAt))
                .orElse(null);
    }

    private TaskAttemptResult toAttemptResult(TaskAttempt attempt) {
        return new TaskAttemptResult(attempt.getAttemptNo(), attempt.getStatus().getCode(),
                attempt.getStartedAt(), attempt.getFinishedAt(), attempt.getFailedAt());
    }

    private TaskEventResult toEventResult(TaskEvent event) {
        return new TaskEventResult(event.getEventType().getCode(), event.getMessage(), event.getOccurredAt());
    }

    private static void requireAdmin(String role) {
        if (!"ADMIN".equals(role)) {
            throw new TaskException(FORBIDDEN_CODE, "无权限");
        }
    }

    private static String sanitize(String value) {
        if (value == null) {
            return null;
        }
        return value
                .replaceAll("(?i)token\\s*=\\s*[^\\s,;]+", "token=***")
                .replaceAll("(?i)secret\\s*=\\s*[^\\s,;]+", "secret=***");
    }

    private static TaskStatusResult statusResultFromIdempotency(String requestHash, TaskIdempotencyRecord record) {
        if (!record.getRequestHash().equals(requestHash)) {
            throw new TaskException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        return TaskIdempotencySnapshotCodec.parseStatusSnapshot(record.getResponseSnapshot());
    }
}
