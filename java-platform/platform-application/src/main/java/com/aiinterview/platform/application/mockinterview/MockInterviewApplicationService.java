package com.aiinterview.platform.application.mockinterview;
import com.aiinterview.platform.application.mockinterview.command.CreateMockInterviewCommand;
import com.aiinterview.platform.application.mockinterview.command.SubmitMockInterviewAnswerCommand;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerFile;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerSubmittedResult;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCurrentState;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCreatedResult;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewResumeFile;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewFileBindingPort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewMemoryCachePort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewResultPort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewTaskCreation;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewTaskPort;
import com.aiinterview.platform.application.file.FileApplicationService;
import com.aiinterview.platform.application.file.command.BindFileToTaskCommand;
import com.aiinterview.platform.application.task.TaskApplicationService;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.task.Task;
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
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;

/**
 * 模拟面试应用服务。
 */
public class MockInterviewApplicationService {

    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;
    private static final int IDEMPOTENCY_HASH_CONFLICT_CODE = 71008;
    private static final int TASK_NOT_FOUND_CODE = 71003;
    private static final int VERSION_CONFLICT_CODE = 71009;
    private static final int TASK_TYPE_INVALID_CODE = 77007;
    private static final int TASK_STATUS_INVALID_CODE = 77008;
    private static final int ROUND_MISMATCH_CODE = 77010;
    private static final String ANSWER_OPERATION = "mock_interview_answer";

    private final MockInterviewTaskPort taskPort;
    private final ObjectMapper objectMapper;
    private final TaskRepository taskRepository;
    private final TaskEventRepository eventRepository;
    private final TaskIdempotencyRepository idempotencyRepository;
    private final MockInterviewResultPort resultPort;
    private final MockInterviewFileBindingPort fileBindingPort;
    private final FileApplicationService fileApplicationService;
    private final TaskCommandOutboxPort outboxPort;
    private final AuditWritePort auditWritePort;
    private final Clock clock;
    private final Supplier<String> idSupplier;
    private final MockInterviewMemoryCachePort memoryCachePort;

    public MockInterviewApplicationService(MockInterviewTaskPort taskPort, ObjectMapper objectMapper) {
        this(taskPort, objectMapper, null, null, null, null, null, null, null, null, null, null, null);
    }

    public MockInterviewApplicationService(
            MockInterviewTaskPort taskPort,
            ObjectMapper objectMapper,
            TaskRepository taskRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            MockInterviewResultPort resultPort,
            MockInterviewFileBindingPort fileBindingPort,
            FileApplicationService fileApplicationService,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier,
            MockInterviewMemoryCachePort memoryCachePort
    ) {
        this.taskPort = Objects.requireNonNull(taskPort, "taskPort 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
        this.taskRepository = taskRepository;
        this.eventRepository = eventRepository;
        this.idempotencyRepository = idempotencyRepository;
        this.resultPort = resultPort;
        this.fileBindingPort = fileBindingPort;
        this.fileApplicationService = fileApplicationService;
        this.outboxPort = outboxPort;
        this.auditWritePort = auditWritePort;
        this.clock = clock;
        this.idSupplier = idSupplier;
        this.memoryCachePort = memoryCachePort;
    }

    public MockInterviewApplicationService(
            MockInterviewTaskPort taskPort,
            ObjectMapper objectMapper,
            TaskRepository taskRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            MockInterviewResultPort resultPort,
            MockInterviewFileBindingPort fileBindingPort,
            FileApplicationService fileApplicationService,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this(
                taskPort,
                objectMapper,
                taskRepository,
                eventRepository,
                idempotencyRepository,
                resultPort,
                fileBindingPort,
                fileApplicationService,
                outboxPort,
                auditWritePort,
                clock,
                idSupplier,
                null
        );
    }

    public MockInterviewCreatedResult createMockInterview(CreateMockInterviewCommand command) {
        Objects.requireNonNull(command, "command 不能为空");
        MockInterviewResumeFile resumeFile = resumeFile(command);
        Map<String, Object> inputSummary = inputSummary(command, resumeFile);
        return taskPort.createMockInterviewTask(new MockInterviewTaskCreation(
                command.userId(),
                command.idempotencyKey(),
                command.requestHash(),
                command.traceId(),
                resumeFile.fileId(),
                toJson(inputSummary),
                workerPayload(command, resumeFile, inputSummary)
        ));
    }

    @Transactional(readOnly = true)
    public MockInterviewCurrentState getCurrentState(String userId, String taskId) {
        requireStateQueryDependencies();
        Task task = taskRepository.findById(taskId)
                .orElseThrow(() -> new MockInterviewException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (!task.getUserId().equals(userId)) {
            auditWritePort.writeAudit("TASK_ACCESS_DENIED", userId, taskId, Map.of("operation", "mock_interview_state"));
            throw new MockInterviewException(TASK_NOT_FOUND_CODE, "任务不存在");
        }
        if (task.getTaskType() != TaskType.MOCK_INTERVIEW) {
            throw new MockInterviewException(TASK_TYPE_INVALID_CODE, "只能查询模拟面试状态");
        }
        return resultPort.findCurrentState(task.getId(), task.getCurrentAttempt())
                .orElseThrow(() -> new MockInterviewException(ROUND_MISMATCH_CODE, "模拟面试等待轮次不存在"));
    }

    @Transactional
    public MockInterviewAnswerSubmittedResult submitAnswer(SubmitMockInterviewAnswerCommand command) {
        requireSubmitDependencies();
        Objects.requireNonNull(command, "command 不能为空");
        TaskIdempotencyRecord existing = idempotencyRepository
                .findByUserIdAndOperationTypeAndIdempotencyKey(
                        command.userId(),
                        ANSWER_OPERATION,
                        command.idempotencyKey()
                )
                .orElse(null);
        if (existing != null) {
            return answerResultFromIdempotency(command, existing);
        }

        Task task = taskRepository.findById(command.taskId())
                .orElseThrow(() -> new MockInterviewException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (!task.getUserId().equals(command.userId())) {
            auditWritePort.writeAudit("TASK_ACCESS_DENIED", command.userId(), command.taskId(), Map.of(
                    "operation", ANSWER_OPERATION
            ));
            throw new MockInterviewException(TASK_NOT_FOUND_CODE, "任务不存在");
        }
        if (task.getTaskType() != TaskType.MOCK_INTERVIEW) {
            throw new MockInterviewException(TASK_TYPE_INVALID_CODE, "只能提交模拟面试回答");
        }
        if (task.getStatus() != TaskStatus.WAITING_USER_INPUT) {
            throw new MockInterviewException(TASK_STATUS_INVALID_CODE, "任务不在等待回答状态");
        }
        MockInterviewCurrentState state = resultPort.findCurrentState(task.getId(), task.getCurrentAttempt())
                .orElseThrow(() -> new MockInterviewException(ROUND_MISMATCH_CODE, "模拟面试等待轮次不存在"));
        if (!state.currentQuestionId().equals(command.questionId()) || !state.currentRoundId().equals(command.roundId())) {
            throw new MockInterviewException(ROUND_MISMATCH_CODE, "模拟面试轮次已变化，请刷新任务详情");
        }

        MockInterviewAnswerFile answerFile = fileBindingPort.validateAnswerFile(command.userId(), command.answerFileId());
        if (!resultPort.markAnswerSubmitted(state, answerFile)) {
            throw new MockInterviewException(VERSION_CONFLICT_CODE, "模拟面试状态版本冲突");
        }
        fileApplicationService.bindFileToTask(new BindFileToTaskCommand(
                command.userId(),
                task.getId(),
                answerFile.fileId(),
                "input",
                task.getCurrentAttempt(),
                command.traceId()
        ));

        TaskStatus fromStatus = task.getStatus();
        long expectedVersion = task.getVersion();
        Instant now = Instant.now(clock);
        task.markQueued(now);
        if (!taskRepository.updateByIdAndVersion(task, expectedVersion)) {
            throw new MockInterviewException(VERSION_CONFLICT_CODE, "任务版本冲突");
        }
        MockInterviewAnswerSubmittedResult result = new MockInterviewAnswerSubmittedResult(
                task.getId(),
                task.getStatus().getCode(),
                "排队中",
                task.getCurrentAttempt()
        );
        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                task.getCurrentAttempt(),
                TaskEventType.TASK_QUEUED,
                fromStatus,
                task.getStatus(),
                "process_answer",
                null,
                "模拟面试回答已提交",
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                command.userId(),
                command.traceId(),
                now,
                now
        ));
        outboxPort.enqueueTaskCommand(task, TaskApplicationService.CREATE_TASK_OPERATION,
                processAnswerPayload(task, state, answerFile, command.traceId()));
        idempotencyRepository.save(toAnswerIdempotencyRecord(command, result, now));
        auditWritePort.writeAudit("MOCK_INTERVIEW_ANSWER_SUBMITTED", command.userId(), task.getId(), Map.of(
                "question_id", command.questionId(),
                "round_id", command.roundId(),
                "trace_id", command.traceId()
        ));
        return result;
    }

    private MockInterviewResumeFile resumeFile(CreateMockInterviewCommand command) {
        if (fileBindingPort != null) {
            return fileBindingPort.validateResumeFile(command.userId(), command.resumeFileId());
        }
        return new MockInterviewResumeFile(
                command.resumeFileId(),
                command.resumeFileId(),
                command.resumeFileId(),
                "pdf",
                "resume",
                "application/pdf"
        );
    }

    private static Map<String, Object> inputSummary(
            CreateMockInterviewCommand command,
            MockInterviewResumeFile resumeFile
    ) {
        Map<String, Object> targetJob = new LinkedHashMap<>();
        targetJob.put("title", command.targetJobTitle());
        targetJob.put("description", command.targetJobDescription());

        Map<String, Object> resumeFilePayload = new LinkedHashMap<>();
        resumeFilePayload.put("file_id", resumeFile.fileId());
        resumeFilePayload.put("file_name", resumeFile.originalName());
        resumeFilePayload.put("file_type", resumeFile.fileType());
        resumeFilePayload.put("file_purpose", resumeFile.filePurpose());
        resumeFilePayload.put("mime_type", resumeFile.mimeType());

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("schema_version", 1);
        summary.put("action", "generate_interview_questions");
        summary.put("target_job", targetJob);
        summary.put("resume_file", resumeFilePayload);
        summary.put("question_count", command.questionCount());
        summary.put("max_follow_up_rounds", command.maxFollowUpRounds());
        if (command.llmConfig() != null) {
            summary.put("llm_config", command.llmConfig().publicPayload());
        }
        return summary;
    }

    private static Map<String, Object> workerPayload(
            CreateMockInterviewCommand command,
            MockInterviewResumeFile resumeFile,
            Map<String, Object> inputSummary
    ) {
        Map<String, Object> inputFile = new LinkedHashMap<>();
        inputFile.put("file_id", resumeFile.fileId());
        inputFile.put("object_key", resumeFile.objectKey());
        inputFile.put("file_type", resumeFile.fileType());
        inputFile.put("file_purpose", resumeFile.filePurpose());
        inputFile.put("mime_type", resumeFile.mimeType());

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("task_type", "mock_interview");
        payload.put("user_id", command.userId());
        payload.put("input_summary", inputSummary);
        payload.put("input_files", List.of(inputFile));
        if (command.llmConfig() != null) {
            payload.put("llm_config", command.llmConfig().privatePayload());
        }
        return payload;
    }

    private Map<String, Object> processAnswerPayload(
            Task task,
            MockInterviewCurrentState state,
            MockInterviewAnswerFile answerFile,
            String traceId
    ) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("task_type", "mock_interview");
        payload.put("user_id", task.getUserId());
        payload.put("input_summary", processInputSummary(task, state, answerFile, traceId));
        payload.put("input_files", List.of(answerFilePayload(answerFile)));
        return payload;
    }

    private Map<String, Object> processInputSummary(
            Task task,
            MockInterviewCurrentState state,
            MockInterviewAnswerFile answerFile,
            String traceId
    ) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("schema_version", 1);
        summary.put("action", "process_answer");
        summary.put("task_id", task.getId());
        summary.put("attempt", task.getCurrentAttempt());
        summary.put("trace_id", traceId);
        summary.put("current_question_id", state.currentQuestionId());
        summary.put("current_round_id", state.currentRoundId());
        summary.put("waiting_input_type", state.waitingInputType());
        summary.put("answer_file", answerFilePayload(answerFile));
        Map<String, Object> statePayload = new LinkedHashMap<>(state.statePayload());
        if (memoryCachePort != null) {
            memoryCachePort.findValid(task.getId(), task.getCurrentAttempt(), state.version())
                    .map(cache -> cache.get("task_memory"))
                    .ifPresent(taskMemory -> statePayload.put("task_memory", taskMemory));
        }
        summary.put("interview_state", statePayload);
        Object llmConfig = statePayload.get("llm_config");
        if (llmConfig != null) {
            summary.put("llm_config", llmConfig);
        }
        return summary;
    }

    private static Map<String, Object> answerFilePayload(MockInterviewAnswerFile answerFile) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("file_id", answerFile.fileId());
        payload.put("object_key", answerFile.objectKey());
        payload.put("file_type", answerFile.fileType());
        payload.put("file_purpose", answerFile.filePurpose());
        payload.put("media_type", answerFile.mediaType());
        payload.put("mime_type", answerFile.mimeType());
        return payload;
    }

    private String toJson(Map<String, Object> value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new MockInterviewException(TASK_COMMAND_WRITE_FAILED_CODE, "任务输入摘要序列化失败", ex);
        }
    }

    private MockInterviewAnswerSubmittedResult answerResultFromIdempotency(
            SubmitMockInterviewAnswerCommand command,
            TaskIdempotencyRecord record
    ) {
        if (!record.getRequestHash().equals(command.requestHash())) {
            throw new MockInterviewException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        try {
            JsonNode root = objectMapper.readTree(record.getResponseSnapshot());
            return new MockInterviewAnswerSubmittedResult(
                    requiredText(root, "task_id"),
                    requiredText(root, "status"),
                    requiredText(root, "status_label"),
                    root.path("attempt").asInt()
            );
        } catch (Exception ex) {
            throw new MockInterviewException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照格式非法", ex);
        }
    }

    private TaskIdempotencyRecord toAnswerIdempotencyRecord(
            SubmitMockInterviewAnswerCommand command,
            MockInterviewAnswerSubmittedResult result,
            Instant now
    ) {
        return new TaskIdempotencyRecord(
                idSupplier.get(),
                command.userId(),
                command.idempotencyKey(),
                ANSWER_OPERATION,
                command.requestHash(),
                result.taskId(),
                answerSnapshot(result),
                now.plusSeconds(86400),
                now,
                now
        );
    }

    private String answerSnapshot(MockInterviewAnswerSubmittedResult result) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("task_id", result.taskId());
        snapshot.put("status", result.status());
        snapshot.put("status_label", result.statusLabel());
        snapshot.put("attempt", result.attempt());
        try {
            return objectMapper.writeValueAsString(snapshot);
        } catch (JsonProcessingException ex) {
            throw new MockInterviewException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照序列化失败", ex);
        }
    }

    private static String requiredText(JsonNode root, String field) {
        JsonNode value = root.get(field);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("缺少字段 " + field);
        }
        return value.asText();
    }

    private void requireSubmitDependencies() {
        if (taskRepository == null
                || eventRepository == null
                || idempotencyRepository == null
                || resultPort == null
                || fileBindingPort == null
                || fileApplicationService == null
                || outboxPort == null
                || auditWritePort == null
                || clock == null
                || idSupplier == null) {
            throw new IllegalStateException("模拟面试回答提交依赖未装配");
        }
    }

    private void requireStateQueryDependencies() {
        if (taskRepository == null || resultPort == null || auditWritePort == null) {
            throw new IllegalStateException("模拟面试状态查询依赖未装配");
        }
    }
}
