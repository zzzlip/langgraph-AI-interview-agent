package com.aiinterview.platform.application.asyncmessage;

import com.aiinterview.platform.application.asyncmessage.command.WorkerFeedbackEventCommand;
import com.aiinterview.platform.application.asyncmessage.dto.WorkerFeedbackInboxResult;
import com.aiinterview.platform.application.asyncmessage.port.SseTaskEventPublisher;
import com.aiinterview.platform.application.task.TaskException;
import com.aiinterview.platform.application.task.TaskWorkerFeedbackService;
import com.aiinterview.platform.application.task.command.WorkerFailureCommand;
import com.aiinterview.platform.application.task.command.WorkerProgressCommand;
import com.aiinterview.platform.application.task.command.WorkerResultCommand;
import com.aiinterview.platform.application.task.dto.WorkerFeedbackResult;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRecord;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRepository;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterSourceType;
import com.aiinterview.platform.domain.asyncmessage.InboxEvent;
import com.aiinterview.platform.domain.asyncmessage.InboxEventRepository;
import com.aiinterview.platform.domain.asyncmessage.WorkerFeedbackType;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.time.Instant;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Worker 反馈消费编排服务。
 */
public class WorkerFeedbackConsumeService {

    private static final int WORKER_EVENT_FORMAT_INVALID_CODE = 74005;
    private static final int WORKER_FEEDBACK_HANDLE_FAILED_CODE = 74009;

    private final WorkerFeedbackInboxService inboxService;
    private final InboxEventRepository inboxEventRepository;
    private final DeadLetterRepository deadLetterRepository;
    private final TaskWorkerFeedbackService taskWorkerFeedbackService;
    private final SseTaskEventPublisher sseTaskEventPublisher;
    private final ObjectMapper objectMapper;
    private final Clock clock;
    private final Supplier<String> idSupplier;

    public WorkerFeedbackConsumeService(
            WorkerFeedbackInboxService inboxService,
            InboxEventRepository inboxEventRepository,
            DeadLetterRepository deadLetterRepository,
            TaskWorkerFeedbackService taskWorkerFeedbackService,
            SseTaskEventPublisher sseTaskEventPublisher,
            ObjectMapper objectMapper,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this.inboxService = Objects.requireNonNull(inboxService, "inboxService 不能为空");
        this.inboxEventRepository = Objects.requireNonNull(inboxEventRepository, "inboxEventRepository 不能为空");
        this.deadLetterRepository = Objects.requireNonNull(deadLetterRepository, "deadLetterRepository 不能为空");
        this.taskWorkerFeedbackService = Objects.requireNonNull(taskWorkerFeedbackService, "taskWorkerFeedbackService 不能为空");
        this.sseTaskEventPublisher = Objects.requireNonNull(sseTaskEventPublisher, "sseTaskEventPublisher 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
    }

    public WorkerFeedbackInboxResult consume(WorkerFeedbackEventCommand command) {
        WorkerFeedbackInboxResult inboxResult = inboxService.recordReceived(command);
        if (inboxResult.status() == WorkerFeedbackInboxResult.Status.DUPLICATE_ACK) {
            return inboxResult;
        }

        InboxEvent processing = inboxResult.event().markProcessing(now());
        inboxEventRepository.update(processing);
        try {
            WorkerFeedbackResult feedbackResult = dispatch(command, processing);
            InboxEvent completed = feedbackResult.ignored()
                    ? processing.markIgnored("spec03 忽略 Worker 反馈", now())
                    : processing.markProcessed(now());
            inboxEventRepository.update(completed);
            if (processing.getFeedbackType() != WorkerFeedbackType.CONSUMED) {
                sseTaskEventPublisher.publishTaskChanged(processing.getTaskId());
            }
            return WorkerFeedbackInboxResult.received(completed);
        } catch (TaskException ex) {
            InboxEvent failed = processing.markFailed("spec03 feedback failed: " + ex.getCode(), now());
            inboxEventRepository.update(failed);
            throw new AsyncMessageException(WORKER_FEEDBACK_HANDLE_FAILED_CODE, "Worker 反馈处理失败", ex);
        } catch (RuntimeException ex) {
            InboxEvent failed = processing.markFailed("worker feedback failed", now());
            inboxEventRepository.update(failed);
            throw new AsyncMessageException(WORKER_FEEDBACK_HANDLE_FAILED_CODE, "Worker 反馈处理失败", ex);
        }
    }

    private WorkerFeedbackResult dispatch(WorkerFeedbackEventCommand command, InboxEvent inboxEvent) {
        JsonNode root = read(command.rawPayload());
        JsonNode payload = root.path("payload");
        return switch (inboxEvent.getFeedbackType()) {
            case CONSUMED -> new WorkerFeedbackResult(inboxEvent.getTaskId(), "consumed", false);
            case PROGRESS -> taskWorkerFeedbackService.handleProgress(new WorkerProgressCommand(
                    inboxEvent.getTaskId(),
                    inboxEvent.getAttemptNo(),
                    parseTaskStatus(requiredText(payload, "status")),
                    optionalText(payload, "stage"),
                    optionalInt(payload, "progress"),
                    optionalText(payload, "message"),
                    inboxEvent.getTraceId()
            ));
            case PROBLEM_SET -> throw new AsyncMessageException(
                    WORKER_EVENT_FORMAT_INVALID_CODE,
                    "算法题单反馈必须走专用处理入口"
            );
            case MOCK_INTERVIEW_INTERMEDIATE -> throw new AsyncMessageException(
                    WORKER_EVENT_FORMAT_INVALID_CODE,
                    "模拟面试中间反馈必须走专用处理入口"
            );
            case RESULT -> taskWorkerFeedbackService.handleResult(new WorkerResultCommand(
                    inboxEvent.getTaskId(),
                    inboxEvent.getAttemptNo(),
                    payload.toString(),
                    inboxEvent.getTraceId()
            ));
            case FAILURE -> handleFailure(inboxEvent, payload);
        };
    }

    private WorkerFeedbackResult handleFailure(InboxEvent inboxEvent, JsonNode payload) {
        boolean retryable = payload.path("retryable").asBoolean(true);
        WorkerFeedbackResult result = taskWorkerFeedbackService.handleFailure(new WorkerFailureCommand(
                inboxEvent.getTaskId(),
                inboxEvent.getAttemptNo(),
                requiredText(payload, "failed_stage"),
                optionalText(payload, "failure_code") == null ? "WORKER_FAILED" : optionalText(payload, "failure_code"),
                requiredText(payload, "user_message"),
                optionalText(payload, "admin_summary") == null ? "worker failure" : optionalText(payload, "admin_summary"),
                retryable,
                inboxEvent.getTraceId()
        ));
        if (!retryable) {
            saveWorkerFailureDeadLetter(inboxEvent, payload);
        }
        return result;
    }

    private void saveWorkerFailureDeadLetter(InboxEvent inboxEvent, JsonNode payload) {
        Instant now = now();
        DeadLetterRecord record = new DeadLetterRecord(
                idSupplier.get(),
                DeadLetterSourceType.WORKER_FAILURE,
                inboxEvent.getId(),
                inboxEvent.getTaskId(),
                inboxEvent.getTaskType(),
                inboxEvent.getAttemptNo(),
                optionalText(payload, "failed_stage"),
                optionalText(payload, "admin_summary") == null ? "worker non-retryable failure" : optionalText(payload, "admin_summary"),
                false,
                false,
                inboxEvent.getTraceId(),
                now,
                now
        );
        if (deadLetterRepository.findBySource(record.getSourceType(), record.getSourceId()).isEmpty()) {
            deadLetterRepository.save(record);
        }
    }

    private JsonNode read(String rawPayload) {
        try {
            return objectMapper.readTree(rawPayload);
        } catch (Exception ex) {
            throw new AsyncMessageException(WORKER_EVENT_FORMAT_INVALID_CODE, "Worker 事件格式非法", ex);
        }
    }

    private static TaskStatus parseTaskStatus(String status) {
        return Arrays.stream(TaskStatus.values())
                .filter(taskStatus -> taskStatus.getCode().equals(status))
                .findFirst()
                .orElseThrow(() -> new AsyncMessageException(WORKER_EVENT_FORMAT_INVALID_CODE, "Worker 事件格式非法"));
    }

    private static String requiredText(JsonNode node, String field) {
        String value = optionalText(node, field);
        if (value == null) {
            throw new AsyncMessageException(WORKER_EVENT_FORMAT_INVALID_CODE, "Worker 事件格式非法");
        }
        return value;
    }

    private static String optionalText(JsonNode node, String field) {
        JsonNode value = node.path(field);
        return value.isMissingNode() || value.isNull() || value.asText().isBlank() ? null : value.asText();
    }

    private static Integer optionalInt(JsonNode node, String field) {
        JsonNode value = node.path(field);
        return value.isMissingNode() || value.isNull() ? null : value.asInt();
    }

    private Instant now() {
        return Instant.now(clock);
    }
}
