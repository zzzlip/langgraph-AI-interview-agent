package com.aiinterview.platform.application.task;

import com.aiinterview.platform.application.task.command.WorkerFailureCommand;
import com.aiinterview.platform.application.task.command.WorkerProgressCommand;
import com.aiinterview.platform.application.task.command.WorkerResultCommand;
import com.aiinterview.platform.application.task.dto.WorkerFeedbackResult;
import com.aiinterview.platform.application.task.port.BindResultReferenceCommand;
import com.aiinterview.platform.application.task.port.TaskResultReferencePort;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskAttempt;
import com.aiinterview.platform.domain.task.TaskAttemptRepository;
import com.aiinterview.platform.domain.task.TaskEvent;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskEventType;
import com.aiinterview.platform.domain.task.TaskFailure;
import com.aiinterview.platform.domain.task.TaskOperatorType;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.domain.task.TaskStatus;
import java.time.Clock;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Worker 反馈处理服务。
 */
public class TaskWorkerFeedbackService {

    private static final int TASK_NOT_FOUND_CODE = 71003;
    private static final int ATTEMPT_MISMATCH_CODE = 71011;
    private static final int FEEDBACK_STATUS_INVALID_CODE = 71012;
    private static final int VERSION_CONFLICT_CODE = 71009;

    private final TaskRepository taskRepository;
    private final TaskAttemptRepository attemptRepository;
    private final TaskEventRepository eventRepository;
    private final TaskResultReferencePort resultReferencePort;
    private final Clock clock;
    private final Supplier<String> idSupplier;
    private final ObjectMapper objectMapper;

    public TaskWorkerFeedbackService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskResultReferencePort resultReferencePort,
            Clock clock,
            Supplier<String> idSupplier,
            ObjectMapper objectMapper
    ) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.attemptRepository = Objects.requireNonNull(attemptRepository, "attemptRepository 不能为空");
        this.eventRepository = Objects.requireNonNull(eventRepository, "eventRepository 不能为空");
        this.resultReferencePort = Objects.requireNonNull(resultReferencePort, "resultReferencePort 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    @Transactional
    public WorkerFeedbackResult handleProgress(WorkerProgressCommand command) {
        Task task = findTask(command.taskId());
        AttemptCheck check = checkAttempt(task, command.attempt(), command.traceId());
        if (check.ignored()) {
            return ignoredResult(task);
        }
        if (task.getStatus().isFinal()) {
            appendIgnored(task, command.attempt(), command.traceId(), "终态任务忽略进度反馈");
            return ignoredResult(task);
        }

        TaskStatus fromStatus = task.getStatus();
        Instant now = Instant.now(clock);
        if (command.targetStatus() == TaskStatus.RUNNING) {
            task.markRunning(command.stage(), command.progress(), now);
            updateTask(task, fromStatus);
            appendEvent(task, command.attempt(), TaskEventType.TASK_PROGRESS_UPDATED, fromStatus, task.getStatus(),
                    command.stage(), command.progress(), command.message(), command.traceId(), now);
            return activeResult(task);
        }
        if (command.targetStatus() == TaskStatus.WAITING_USER_INPUT) {
            task.markWaitingUserInput(command.stage(), command.message(), now);
            updateTask(task, fromStatus);
            appendEvent(task, command.attempt(), TaskEventType.TASK_WAITING_USER_INPUT, fromStatus, task.getStatus(),
                    command.stage(), command.progress(), command.message(), command.traceId(), now);
            return activeResult(task);
        }
        if (command.targetStatus() == TaskStatus.QUEUED) {
            task.markQueued(now);
            updateTask(task, fromStatus);
            appendEvent(task, command.attempt(), TaskEventType.TASK_QUEUED, fromStatus, task.getStatus(),
                    command.stage(), command.progress(), command.message(), command.traceId(), now);
            return activeResult(task);
        }
        throw new TaskException(FEEDBACK_STATUS_INVALID_CODE, "Worker 反馈状态不合法");
    }

    @Transactional
    public WorkerFeedbackResult handleResult(WorkerResultCommand command) {
        Task task = findTask(command.taskId());
        AttemptCheck check = checkAttempt(task, command.attempt(), command.traceId());
        if (check.ignored() || task.getStatus().isFinal()) {
            appendIgnored(task, command.attempt(), command.traceId(), "终态或过期结果反馈被忽略");
            return ignoredResult(task);
        }

        TaskStatus fromStatus = task.getStatus();
        Instant now = Instant.now(clock);
        String resultReferenceId;
        try {
            resultReferenceId = resultReferencePort.bindResultReference(toBindResultCommand(task, command));
        } catch (RuntimeException ex) {
            TaskFailure failure = new TaskFailure("result_bind", "RESULT_BIND_FAILED",
                    "结果保存失败，请稍后重试", "result reference bind failed", true, now);
            task.markFailed(failure, now);
            updateTask(task, fromStatus);
            updateAttemptFailed(task, command.attempt(), failure, now);
            appendEvent(task, command.attempt(), TaskEventType.TASK_FAILED, fromStatus, task.getStatus(),
                    task.getStage(), task.getProgress(), failure.getUserMessage(), command.traceId(), now);
            return activeResult(task);
        }

        task.markSuccess(resultReferenceId, now);
        updateTask(task, fromStatus);
        TaskAttempt attempt = currentAttempt(task, command.attempt())
                .orElse(new TaskAttempt(idSupplier.get(), task.getId(), command.attempt(), TaskStatus.SUCCESS,
                        null, now, null, null, null, null, null, now, now));
        attemptRepository.update(new TaskAttempt(attempt.getId(), task.getId(), command.attempt(), TaskStatus.SUCCESS,
                attempt.getStartedAt(), now, null, null, null, null, null, attempt.getCreatedAt(), now));
        appendEvent(task, command.attempt(), TaskEventType.TASK_SUCCEEDED, fromStatus, task.getStatus(),
                task.getStage(), task.getProgress(), "任务成功", command.traceId(), now);
        return activeResult(task);
    }

    @Transactional
    public WorkerFeedbackResult handleFailure(WorkerFailureCommand command) {
        Task task = findTask(command.taskId());
        AttemptCheck check = checkAttempt(task, command.attempt(), command.traceId());
        if (check.ignored() || task.getStatus().isFinal()) {
            appendIgnored(task, command.attempt(), command.traceId(), "终态或过期失败反馈被忽略");
            return ignoredResult(task);
        }

        TaskStatus fromStatus = task.getStatus();
        Instant now = Instant.now(clock);
        TaskFailure failure = new TaskFailure(command.failureStage(), command.failureCode(),
                command.userMessage(), command.adminSummary(), command.retryable(), now);
        task.markFailed(failure, now);
        updateTask(task, fromStatus);
        updateAttemptFailed(task, command.attempt(), failure, now);
        appendEvent(task, command.attempt(), TaskEventType.TASK_FAILED, fromStatus, task.getStatus(),
                command.failureStage(), task.getProgress(), command.userMessage(), command.traceId(), now);
        return activeResult(task);
    }

    private Task findTask(String taskId) {
        return taskRepository.findById(taskId)
                .orElseThrow(() -> new TaskException(TASK_NOT_FOUND_CODE, "任务不存在"));
    }

    private AttemptCheck checkAttempt(Task task, int attemptNo, String traceId) {
        if (attemptNo < task.getCurrentAttempt()) {
            appendIgnored(task, attemptNo, traceId, "过期 attempt 反馈被忽略");
            return new AttemptCheck(true);
        }
        if (attemptNo > task.getCurrentAttempt()) {
            throw new TaskException(ATTEMPT_MISMATCH_CODE, "Worker 反馈 attempt 不匹配");
        }
        return new AttemptCheck(false);
    }

    private java.util.Optional<TaskAttempt> currentAttempt(Task task, int attemptNo) {
        return attemptRepository.findByTaskIdAndAttemptNo(task.getId(), attemptNo);
    }

    private void updateAttemptFailed(Task task, int attemptNo, TaskFailure failure, Instant now) {
        currentAttempt(task, attemptNo).ifPresent(attempt -> attemptRepository.update(attempt.markFailed(
                failure.getFailureStage(),
                failure.getFailureCode(),
                failure.getUserMessage(),
                failure.getAdminSummary(),
                now
        )));
    }

    private void updateTask(Task task, TaskStatus expectedFromStatus) {
        if (!taskRepository.updateByIdAndVersion(task, task.getVersion())) {
            throw new TaskException(VERSION_CONFLICT_CODE, "任务版本冲突");
        }
    }

    private BindResultReferenceCommand toBindResultCommand(Task task, WorkerResultCommand command) {
        JsonNode payload = resultPayload(command.resultPayload());
        JsonNode summary = requiredObject(payload, "result_summary");
        String resultStatus = resultStatus(payload);
        String resultType = optionalText(payload, "result_type");
        if (resultType == null) {
            resultType = requiredText(summary, "result_type");
        }
        return new BindResultReferenceCommand(
                task.getId(),
                task.getUserId(),
                command.attempt(),
                resultStatus,
                resultType,
                summary.toString(),
                outputFiles(payload.path("output_files")),
                command.traceId()
        );
    }

    private JsonNode resultPayload(String rawPayload) {
        try {
            JsonNode root = objectMapper.readTree(rawPayload);
            JsonNode payload = root.path("payload");
            return payload.isObject() ? payload : root;
        } catch (Exception ex) {
            throw new TaskException(FEEDBACK_STATUS_INVALID_CODE, "Worker 结果格式不合法", ex);
        }
    }

    private static String resultStatus(JsonNode payload) {
        String resultStatus = optionalText(payload, "result_status");
        if (resultStatus != null) {
            return resultStatus;
        }
        String status = optionalText(payload, "status");
        if ("success".equals(status)) {
            return "completed";
        }
        if (status != null) {
            return status;
        }
        throw new TaskException(FEEDBACK_STATUS_INVALID_CODE, "Worker 结果格式不合法");
    }

    private static java.util.List<BindResultReferenceCommand.OutputFilePayload> outputFiles(JsonNode outputFiles) {
        if (outputFiles.isMissingNode() || outputFiles.isNull()) {
            return java.util.List.of();
        }
        if (!outputFiles.isArray()) {
            throw new TaskException(FEEDBACK_STATUS_INVALID_CODE, "Worker 结果格式不合法");
        }
        java.util.List<BindResultReferenceCommand.OutputFilePayload> result = new java.util.ArrayList<>();
        for (JsonNode outputFile : outputFiles) {
            result.add(new BindResultReferenceCommand.OutputFilePayload(
                    requiredText(outputFile, "object_key"),
                    requiredText(outputFile, "file_type"),
                    requiredText(outputFile, "file_purpose"),
                    requiredText(outputFile, "mime_type"),
                    requiredPositiveLong(outputFile, "file_size"),
                    requiredText(outputFile, "display_name")
            ));
        }
        return result;
    }

    private static JsonNode requiredObject(JsonNode node, String field) {
        JsonNode value = node.path(field);
        if (!value.isObject()) {
            throw new TaskException(FEEDBACK_STATUS_INVALID_CODE, "Worker 结果格式不合法");
        }
        return value;
    }

    private static String requiredText(JsonNode node, String field) {
        String value = optionalText(node, field);
        if (value == null) {
            throw new TaskException(FEEDBACK_STATUS_INVALID_CODE, "Worker 结果格式不合法");
        }
        return value;
    }

    private static String optionalText(JsonNode node, String field) {
        JsonNode value = node.path(field);
        return value.isMissingNode() || value.isNull() || !value.isValueNode() || value.asText().isBlank() ? null : value.asText();
    }

    private static long requiredPositiveLong(JsonNode node, String field) {
        JsonNode value = node.path(field);
        if (!value.isIntegralNumber() || value.asLong() <= 0) {
            throw new TaskException(FEEDBACK_STATUS_INVALID_CODE, "Worker 结果格式不合法");
        }
        return value.asLong();
    }

    private void appendIgnored(Task task, int attemptNo, String traceId, String message) {
        appendEvent(task, attemptNo, TaskEventType.TASK_FEEDBACK_IGNORED, task.getStatus(), task.getStatus(),
                task.getStage(), task.getProgress(), message, traceId, Instant.now(clock));
    }

    private void appendEvent(
            Task task,
            int attemptNo,
            TaskEventType eventType,
            TaskStatus fromStatus,
            TaskStatus toStatus,
            String stage,
            Integer progress,
            String message,
            String traceId,
            Instant now
    ) {
        eventRepository.save(new TaskEvent(idSupplier.get(), task.getId(), attemptNo, eventType, fromStatus, toStatus,
                stage, progress, message, Map.of("schema_version", 1), TaskOperatorType.WORKER,
                "worker", traceId, now, now));
    }

    private static WorkerFeedbackResult activeResult(Task task) {
        return new WorkerFeedbackResult(task.getId(), task.getStatus().getCode(), false);
    }

    private static WorkerFeedbackResult ignoredResult(Task task) {
        return new WorkerFeedbackResult(task.getId(), task.getStatus().getCode(), true);
    }

    private record AttemptCheck(boolean ignored) {
    }
}
