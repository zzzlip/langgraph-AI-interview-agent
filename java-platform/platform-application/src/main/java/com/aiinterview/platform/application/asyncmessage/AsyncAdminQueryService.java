package com.aiinterview.platform.application.asyncmessage;

import com.aiinterview.platform.application.asyncmessage.dto.AsyncTaskTraceResult;
import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterHandledResult;
import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterItemResult;
import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterPageResult;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRecord;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRepository;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterSourceType;
import com.aiinterview.platform.domain.asyncmessage.InboxEvent;
import com.aiinterview.platform.domain.asyncmessage.InboxEventRepository;
import com.aiinterview.platform.domain.asyncmessage.OutboxEvent;
import com.aiinterview.platform.domain.asyncmessage.OutboxEventRepository;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskRepository;
import java.time.Clock;
import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * 管理员异步协作排查应用服务。
 */
public class AsyncAdminQueryService {

    private static final int DEAD_LETTER_NOT_FOUND_CODE = 74011;
    private static final int DEAD_LETTER_HANDLED_CODE = 74012;

    private final TaskRepository taskRepository;
    private final OutboxEventRepository outboxRepository;
    private final InboxEventRepository inboxRepository;
    private final DeadLetterRepository deadLetterRepository;
    private final AuditWritePort auditWritePort;
    private final Clock clock;

    public AsyncAdminQueryService(TaskRepository taskRepository, OutboxEventRepository outboxRepository,
            InboxEventRepository inboxRepository, DeadLetterRepository deadLetterRepository,
            AuditWritePort auditWritePort, Clock clock) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.outboxRepository = Objects.requireNonNull(outboxRepository, "outboxRepository 不能为空");
        this.inboxRepository = Objects.requireNonNull(inboxRepository, "inboxRepository 不能为空");
        this.deadLetterRepository = Objects.requireNonNull(deadLetterRepository, "deadLetterRepository 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
    }

    public AsyncTaskTraceResult getTaskTrace(String adminId, String taskId) {
        auditWritePort.writeAudit("ASYNC_TRACE_VIEWED", adminId, taskId, Map.of("scope", "task_async_trace"));
        Optional<Task> task = taskRepository.findById(taskId);
        int attempt = task.map(Task::getCurrentAttempt).orElse(1);
        return new AsyncTaskTraceResult(
                taskId,
                task.map(taskValue -> taskValue.getTaskType().getCode()).orElse(null),
                task.map(Task::getCurrentAttempt).orElse(0),
                latestDelivery(taskId),
                latestConsumption(taskId, attempt),
                latestFeedback(taskId, attempt),
                latestDeadLetter(taskId)
        );
    }

    public DeadLetterPageResult listDeadLetters(String adminId, DeadLetterSourceType sourceType, Boolean handled,
            int page, int pageSize) {
        auditWritePort.writeAudit("ASYNC_DEAD_LETTER_VIEWED", adminId, "async_dead_letter",
                Map.of("page", page, "page_size", pageSize));
        List<DeadLetterItemResult> items = deadLetterRepository.findPage(sourceType, handled, page, pageSize)
                .stream()
                .map(this::toItem)
                .toList();
        return new DeadLetterPageResult(items, deadLetterRepository.count(sourceType, handled), page, pageSize);
    }

    public DeadLetterHandledResult markHandled(String adminId, String deadLetterId) {
        DeadLetterRecord record = deadLetterRepository.findById(deadLetterId)
                .orElseThrow(() -> new AsyncMessageException(DEAD_LETTER_NOT_FOUND_CODE, "死信记录不存在"));
        DeadLetterRecord handled;
        try {
            handled = record.markHandled(Instant.now(clock));
        } catch (IllegalStateException ex) {
            throw new AsyncMessageException(DEAD_LETTER_HANDLED_CODE, "死信记录已处理");
        }
        deadLetterRepository.update(handled);
        auditWritePort.writeAudit("ASYNC_DEAD_LETTER_HANDLED", adminId, deadLetterId,
                Map.of("source_type", record.getSourceType().getCode()));
        return new DeadLetterHandledResult(deadLetterId, true);
    }

    private AsyncTaskTraceResult.DeliverySummary latestDelivery(String taskId) {
        return outboxRepository.findByTaskId(taskId).stream()
                .max(Comparator.comparing(OutboxEvent::getCreatedAt))
                .map(event -> new AsyncTaskTraceResult.DeliverySummary(event.getStatus().getCode(),
                        event.getRetryCount(), sanitize(event.getLastError()), event.getCreatedAt(), event.getDeliveredAt()))
                .orElse(null);
    }

    private AsyncTaskTraceResult.ConsumptionSummary latestConsumption(String taskId, int attempt) {
        return inboxRepository.findConsumedByTaskIdAndAttemptNo(taskId, attempt)
                .map(event -> new AsyncTaskTraceResult.ConsumptionSummary(event.getStatus().getCode(),
                        event.getReceivedAt(), event.getProcessedAt()))
                .orElse(null);
    }

    private AsyncTaskTraceResult.FeedbackSummary latestFeedback(String taskId, int attempt) {
        return inboxRepository.findByTaskIdAndAttemptNo(taskId, attempt).stream()
                .max(Comparator.comparing(InboxEvent::getReceivedAt))
                .map(event -> new AsyncTaskTraceResult.FeedbackSummary(event.getEventType(), event.getStatus().getCode(),
                        summaryValue(event.getPayloadSummary(), "stage"),
                        parseInteger(summaryValue(event.getPayloadSummary(), "progress")), event.getReceivedAt()))
                .orElse(null);
    }

    private AsyncTaskTraceResult.DeadLetterSummary latestDeadLetter(String taskId) {
        return deadLetterRepository.findByTaskId(taskId).stream()
                .max(Comparator.comparing(DeadLetterRecord::getCreatedAt))
                .map(record -> new AsyncTaskTraceResult.DeadLetterSummary(record.getId(),
                        record.getSourceType().getCode(), record.getFailureStage(), sanitize(record.getFailureReason()),
                        record.isRetryable(), record.isManualHandled(), record.getCreatedAt()))
                .orElse(null);
    }

    private DeadLetterItemResult toItem(DeadLetterRecord record) {
        return new DeadLetterItemResult(record.getId(), record.getSourceType().getCode(), record.getTaskId(),
                record.getTaskType(), record.getAttemptNo(), record.getFailureStage(), sanitize(record.getFailureReason()),
                record.isRetryable(), record.isManualHandled(), record.getCreatedAt());
    }

    private static Integer parseInteger(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ex) {
            return null;
        }
    }

    private static String summaryValue(String summary, String key) {
        if (summary == null || summary.isBlank()) {
            return null;
        }
        String quotedKey = "\"" + key + "\"";
        int keyIndex = summary.indexOf(quotedKey);
        if (keyIndex < 0) {
            return null;
        }
        int colon = summary.indexOf(':', keyIndex + quotedKey.length());
        if (colon < 0) {
            return null;
        }
        int start = colon + 1;
        while (start < summary.length() && Character.isWhitespace(summary.charAt(start))) {
            start++;
        }
        if (start < summary.length() && summary.charAt(start) == '"') {
            int end = summary.indexOf('"', start + 1);
            return end < 0 ? null : summary.substring(start + 1, end);
        }
        int end = start;
        while (end < summary.length() && ",}".indexOf(summary.charAt(end)) < 0) {
            end++;
        }
        return summary.substring(start, end).trim();
    }

    private static String sanitize(String value) {
        if (value == null) {
            return null;
        }
        return value
                .replaceAll("(?i)object_key[^,}]*", "object_key=<redacted>")
                .replaceAll("(?i)(password|secret|token|session)[^,}]*", "$1=<redacted>");
    }
}
