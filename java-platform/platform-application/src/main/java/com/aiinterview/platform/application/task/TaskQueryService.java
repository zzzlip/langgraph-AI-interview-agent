package com.aiinterview.platform.application.task;

import com.aiinterview.platform.application.result.ResultQueryService;
import com.aiinterview.platform.application.result.dto.ResultDetailResult;
import com.aiinterview.platform.application.task.dto.TaskAttemptResult;
import com.aiinterview.platform.application.task.dto.TaskDetailResult;
import com.aiinterview.platform.application.task.dto.TaskEventResult;
import com.aiinterview.platform.application.task.dto.TaskFailureResult;
import com.aiinterview.platform.application.task.dto.TaskListItem;
import com.aiinterview.platform.application.task.dto.TaskPageResult;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.domain.task.PageResult;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskAttempt;
import com.aiinterview.platform.domain.task.TaskAttemptRepository;
import com.aiinterview.platform.domain.task.TaskEvent;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskQueryCriteria;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.aiinterview.platform.domain.task.TaskType;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 任务查询应用服务。
 */
public class TaskQueryService {

    private static final int TASK_NOT_FOUND_CODE = 71003;

    private final TaskRepository taskRepository;
    private final TaskAttemptRepository attemptRepository;
    private final TaskEventRepository eventRepository;
    private final AuditWritePort auditWritePort;
    private final ResultQueryService resultQueryService;

    public TaskQueryService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            AuditWritePort auditWritePort
    ) {
        this(taskRepository, attemptRepository, eventRepository, auditWritePort, null);
    }

    public TaskQueryService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            AuditWritePort auditWritePort,
            ResultQueryService resultQueryService
    ) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.attemptRepository = Objects.requireNonNull(attemptRepository, "attemptRepository 不能为空");
        this.eventRepository = Objects.requireNonNull(eventRepository, "eventRepository 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.resultQueryService = resultQueryService;
    }

    /**
     * 查询当前用户任务列表。
     */
    public TaskPageResult getTaskList(String userId, TaskType taskType, TaskStatus status, int page, int pageSize) {
        PageResult<Task> result = taskRepository.findByUser(new TaskQueryCriteria(userId, taskType, status, page, pageSize));
        List<TaskListItem> items = result.items().stream()
                .map(this::toListItem)
                .toList();
        return new TaskPageResult(items, result.total(), result.page(), result.pageSize());
    }

    /**
     * 查询当前用户任务详情；他人任务统一返回不存在语义。
     */
    public TaskDetailResult getTaskDetail(String userId, String taskId) {
        Task task = taskRepository.findById(taskId)
                .orElseThrow(() -> new TaskException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (!task.getUserId().equals(userId)) {
            auditWritePort.writeAudit("TASK_ACCESS_DENIED", userId, taskId, Map.of("operation", "task_detail"));
            throw new TaskException(TASK_NOT_FOUND_CODE, "任务不存在");
        }

        List<TaskAttempt> attempts = attemptRepository.findByTaskId(taskId);
        List<TaskEvent> events = eventRepository.findByTaskIdOrderByOccurredAt(taskId);
        ResultDetailResult resultDetail = resultQueryService == null
                ? null
                : resultQueryService.attachResultToTaskDetail(
                        task.getId(),
                        task.getTaskType().getCode(),
                        task.getResultReferenceId(),
                        task.getStatus().getCode()
                );

        return new TaskDetailResult(
                task.getId(),
                task.getTaskType().getCode(),
                task.getStatus().getCode(),
                statusLabel(task.getStatus()),
                task.getStage(),
                task.getProgress(),
                task.getCurrentAttempt(),
                task.getRetryCount(),
                task.getMaxRetryCount(),
                task.isRetryable(),
                task.isCancellable(),
                task.getInputSummary(),
                task.getResultReferenceId(),
                resultDetail,
                toFailureResult(attempts),
                attempts.stream().map(this::toAttemptResult).toList(),
                events.stream().map(this::toEventResult).toList(),
                task.getCreatedAt(),
                task.getUpdatedAt()
        );
    }

    private TaskListItem toListItem(Task task) {
        return new TaskListItem(
                task.getId(),
                task.getTaskType().getCode(),
                task.getStatus().getCode(),
                statusLabel(task.getStatus()),
                task.getStage(),
                task.getProgress(),
                task.isRetryable(),
                task.getRetryCount(),
                task.getMaxRetryCount(),
                task.isCancellable(),
                task.getResultReferenceId() != null,
                task.getCreatedAt(),
                task.getUpdatedAt(),
                task.getCompletedAt(),
                task.getCancelledAt()
        );
    }

    private TaskAttemptResult toAttemptResult(TaskAttempt attempt) {
        return new TaskAttemptResult(
                attempt.getAttemptNo(),
                attempt.getStatus().getCode(),
                attempt.getStartedAt(),
                attempt.getFinishedAt(),
                attempt.getFailedAt()
        );
    }

    private TaskEventResult toEventResult(TaskEvent event) {
        return new TaskEventResult(event.getEventType().getCode(), event.getMessage(), event.getOccurredAt());
    }

    private TaskFailureResult toFailureResult(List<TaskAttempt> attempts) {
        return attempts.stream()
                .filter(attempt -> attempt.getFailedAt() != null)
                .max(Comparator.comparing(TaskAttempt::getFailedAt))
                .map(attempt -> new TaskFailureResult(
                        attempt.getFailureStage(),
                        attempt.getUserMessage(),
                        attempt.getFailedAt()
                ))
                .orElse(null);
    }

    private static String statusLabel(TaskStatus status) {
        return switch (status) {
            case PENDING -> "待处理";
            case QUEUED -> "排队中";
            case RUNNING -> "处理中";
            case WAITING_USER_INPUT -> "等待用户补充";
            case SUCCESS -> "已完成";
            case FAILED -> "失败";
            case CANCELLED -> "已取消";
        };
    }
}
