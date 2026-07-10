package com.aiinterview.platform.application.asyncmessage;

import com.aiinterview.platform.application.asyncmessage.command.WorkerFeedbackEventCommand;
import com.aiinterview.platform.application.asyncmessage.dto.WorkerFeedbackInboxResult;
import com.aiinterview.platform.domain.asyncmessage.AsyncConsumptionStatus;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRecord;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRepository;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterSourceType;
import com.aiinterview.platform.domain.asyncmessage.InboxEvent;
import com.aiinterview.platform.domain.asyncmessage.InboxEventRepository;
import com.aiinterview.platform.domain.asyncmessage.WorkerFeedbackType;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Clock;
import java.time.Instant;
import java.util.HexFormat;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Worker 反馈 Inbox 幂等前置服务。
 */
public class WorkerFeedbackInboxService {

    private static final int WORKER_EVENT_FORMAT_INVALID_CODE = 74005;
    private static final int WORKER_EVENT_ID_CONFLICT_CODE = 74007;
    private static final int INBOX_WRITE_FAILED_CODE = 74008;
    private static final int DEFAULT_MAX_RETRY_COUNT = 5;
    private static final int MAX_PAYLOAD_BYTES = 256 * 1024;
    private static final Set<Integer> SUPPORTED_SCHEMA_VERSIONS = Set.of(1);
    private static final Map<String, WorkerFeedbackType> EVENT_TYPE_MAPPING = Map.of(
            "task.command.consumed", WorkerFeedbackType.CONSUMED,
            "task.progress.updated", WorkerFeedbackType.PROGRESS,
            "algorithm_test.problem_set.generated", WorkerFeedbackType.PROBLEM_SET,
            "mock_interview.questions.generated", WorkerFeedbackType.MOCK_INTERVIEW_INTERMEDIATE,
            "mock_interview.follow_up.generated", WorkerFeedbackType.MOCK_INTERVIEW_INTERMEDIATE,
            "mock_interview.question.completed", WorkerFeedbackType.MOCK_INTERVIEW_INTERMEDIATE,
            "mock_interview.answer.needs_resubmit", WorkerFeedbackType.MOCK_INTERVIEW_INTERMEDIATE,
            "task.result.generated", WorkerFeedbackType.RESULT,
            "task.failed", WorkerFeedbackType.FAILURE
    );

    private final InboxEventRepository inboxEventRepository;
    private final DeadLetterRepository deadLetterRepository;
    private final Clock clock;
    private final Supplier<String> idSupplier;

    public WorkerFeedbackInboxService(
            InboxEventRepository inboxEventRepository,
            DeadLetterRepository deadLetterRepository,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this.inboxEventRepository = Objects.requireNonNull(inboxEventRepository, "inboxEventRepository 不能为空");
        this.deadLetterRepository = Objects.requireNonNull(deadLetterRepository, "deadLetterRepository 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
    }

    public WorkerFeedbackInboxResult recordReceived(WorkerFeedbackEventCommand command) {
        validate(command);
        String payloadHash = sha256(command.rawPayload());
        return inboxEventRepository.findByEventId(command.eventId())
                .map(existing -> handleExisting(existing, command, payloadHash))
                .orElseGet(() -> saveNewInboxEvent(command, payloadHash));
    }

    private WorkerFeedbackInboxResult handleExisting(
            InboxEvent existing,
            WorkerFeedbackEventCommand command,
            String payloadHash
    ) {
        if (existing.hasSamePayloadHash(payloadHash)) {
            return WorkerFeedbackInboxResult.duplicateAck(existing);
        }

        saveConflictDeadLetter(existing, command);
        throw new AsyncMessageException(WORKER_EVENT_ID_CONFLICT_CODE, "Worker 事件 ID 冲突");
    }

    private WorkerFeedbackInboxResult saveNewInboxEvent(WorkerFeedbackEventCommand command, String payloadHash) {
        Instant now = Instant.now(clock);
        InboxEvent event = new InboxEvent(
                idSupplier.get(),
                command.eventId(),
                command.eventType(),
                command.taskId(),
                command.taskType(),
                command.attemptNo(),
                EVENT_TYPE_MAPPING.get(command.eventType()),
                AsyncConsumptionStatus.RECEIVED,
                payloadHash,
                command.payloadSummary(),
                0,
                DEFAULT_MAX_RETRY_COUNT,
                null,
                command.traceId(),
                now,
                null,
                now,
                now
        );

        try {
            inboxEventRepository.save(event);
            return WorkerFeedbackInboxResult.received(event);
        } catch (RuntimeException ex) {
            throw new AsyncMessageException(INBOX_WRITE_FAILED_CODE, "Inbox 写入失败", ex);
        }
    }

    private void saveConflictDeadLetter(InboxEvent existing, WorkerFeedbackEventCommand command) {
        Instant now = Instant.now(clock);
        DeadLetterRecord record = new DeadLetterRecord(
                idSupplier.get(),
                DeadLetterSourceType.INBOX,
                existing.getId(),
                existing.getTaskId(),
                existing.getTaskType(),
                existing.getAttemptNo(),
                "inbox_payload_conflict",
                "相同 event_id 的 Worker 反馈 payload_hash 不一致",
                false,
                false,
                command.traceId(),
                now,
                now
        );
        deadLetterRepository.save(record);
    }

    private static void validate(WorkerFeedbackEventCommand command) {
        if (command == null
                || !SUPPORTED_SCHEMA_VERSIONS.contains(command.schemaVersion())
                || isBlank(command.eventId())
                || isBlank(command.eventType())
                || isBlank(command.taskId())
                || isBlank(command.taskType())
                || command.attemptNo() < 1
                || isBlank(command.traceId())
                || isBlank(command.rawPayload())
                || !EVENT_TYPE_MAPPING.containsKey(command.eventType())
                || command.rawPayload().getBytes(StandardCharsets.UTF_8).length > MAX_PAYLOAD_BYTES) {
            throw new AsyncMessageException(WORKER_EVENT_FORMAT_INVALID_CODE, "Worker 事件格式非法");
        }
    }

    private static String sha256(String payload) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(payload.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 摘要算法不可用", ex);
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
