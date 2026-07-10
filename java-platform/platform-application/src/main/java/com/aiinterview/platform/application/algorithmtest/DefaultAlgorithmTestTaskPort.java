package com.aiinterview.platform.application.algorithmtest;

import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmCheckRequestedResult;
import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmTestCreatedResult;
import com.aiinterview.platform.application.algorithmtest.port.AlgorithmTestCheckRequest;
import com.aiinterview.platform.application.algorithmtest.port.AlgorithmTestTaskCreation;
import com.aiinterview.platform.application.algorithmtest.port.AlgorithmTestTaskPort;
import com.aiinterview.platform.application.task.TaskApplicationService;
import com.aiinterview.platform.application.task.port.AuditWritePort;
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
import com.aiinterview.platform.domain.task.TaskType;
import com.aiinterview.platform.domain.result.ResultType;
import com.aiinterview.platform.domain.result.TaskResult;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;

/**
 * 默认算法测试任务创建端口。
 */
public class DefaultAlgorithmTestTaskPort implements AlgorithmTestTaskPort {

    private static final int IDEMPOTENCY_HASH_CONFLICT_CODE = 71008;
    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;
    private static final int TASK_NOT_FOUND_CODE = 71003;
    private static final int TASK_TYPE_INVALID_CODE = 76005;
    private static final int TASK_STATUS_INVALID_CODE = 76006;
    private static final int PROBLEM_SET_MISSING_CODE = 76007;
    private static final int FINAL_RESULT_EXISTS_CODE = 76008;
    private static final String CHECK_OPERATION = "algorithm_check";

    private final TaskRepository taskRepository;
    private final TaskAttemptRepository attemptRepository;
    private final TaskEventRepository eventRepository;
    private final TaskIdempotencyRepository idempotencyRepository;
    private final TaskResultRepository taskResultRepository;
    private final TaskCommandOutboxPort outboxPort;
    private final AuditWritePort auditWritePort;
    private final Clock clock;
    private final Supplier<String> idSupplier;
    private final ObjectMapper objectMapper;

    public DefaultAlgorithmTestTaskPort(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            TaskResultRepository taskResultRepository,
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
        this.taskResultRepository = Objects.requireNonNull(taskResultRepository, "taskResultRepository 不能为空");
        this.outboxPort = Objects.requireNonNull(outboxPort, "outboxPort 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    @Override
    @Transactional
    public AlgorithmCheckRequestedResult requestAlgorithmCheck(AlgorithmTestCheckRequest request) {
        TaskIdempotencyRecord existing = idempotencyRepository
                .findByUserIdAndOperationTypeAndIdempotencyKey(
                        request.userId(),
                        CHECK_OPERATION,
                        request.idempotencyKey()
                )
                .orElse(null);
        if (existing != null) {
            return checkResultFromIdempotency(request, existing);
        }

        Task task = taskRepository.findById(request.taskId())
                .orElseThrow(() -> new AlgorithmTestException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (!task.getUserId().equals(request.userId())) {
            auditWritePort.writeAudit("TASK_ACCESS_DENIED", request.userId(), request.taskId(), Map.of(
                    "operation", "algorithm_check"
            ));
            throw new AlgorithmTestException(TASK_NOT_FOUND_CODE, "任务不存在");
        }
        if (task.getTaskType() != TaskType.ALGORITHM_TEST) {
            throw new AlgorithmTestException(TASK_TYPE_INVALID_CODE, "只能检查算法测试任务");
        }
        if (hasFinalCheckResult(task.getId())) {
            throw new AlgorithmTestException(FINAL_RESULT_EXISTS_CODE, "最终检查结果已存在");
        }
        if (task.getStatus() != TaskStatus.WAITING_USER_INPUT) {
            throw new AlgorithmTestException(TASK_STATUS_INVALID_CODE, "任务不在等待答题状态");
        }
        TaskResult problemSet = findProblemSet(task.getId())
                .orElseThrow(() -> new AlgorithmTestException(PROBLEM_SET_MISSING_CODE, "题单不存在"));
        Map<String, Object> checkCommandPayload = checkPayload(task, problemSet);

        TaskStatus fromStatus = task.getStatus();
        long expectedVersion = task.getVersion();
        Instant now = Instant.now(clock);
        task.startUserInputFollowupAttempt(now);
        AlgorithmCheckRequestedResult result = new AlgorithmCheckRequestedResult(
                task.getId(),
                task.getStatus().getCode(),
                "排队中",
                task.getCurrentAttempt()
        );

        if (!taskRepository.updateByIdAndVersion(task, expectedVersion)) {
            throw new AlgorithmTestException(71009, "任务版本冲突");
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
                TaskEventType.TASK_QUEUED,
                fromStatus,
                TaskStatus.QUEUED,
                null,
                null,
                "算法测试提交检查已排队",
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                request.userId(),
                request.traceId(),
                now,
                now
        ));
        outboxPort.enqueueTaskCommand(task, TaskApplicationService.CREATE_TASK_OPERATION, checkCommandPayload);
        idempotencyRepository.save(toCheckIdempotencyRecord(request, task.getId(), result, now));
        auditWritePort.writeAudit("ALGORITHM_CHECK_REQUESTED", request.userId(), task.getId(), Map.of(
                "attempt", task.getCurrentAttempt(),
                "trace_id", request.traceId()
        ));
        return result;
    }

    @Override
    @Transactional
    public AlgorithmTestCreatedResult createAlgorithmTestTask(AlgorithmTestTaskCreation creation) {
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
                TaskType.ALGORITHM_TEST,
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
                "算法测试任务已创建",
                Map.of("schema_version", 1),
                TaskOperatorType.USER,
                creation.userId(),
                creation.traceId(),
                now,
                now
        );
        AlgorithmTestCreatedResult result = new AlgorithmTestCreatedResult(
                task.getId(),
                task.getTaskType().getCode(),
                task.getStatus().getCode(),
                "排队中",
                task.getCreatedAt()
        );

        try {
            taskRepository.save(task);
            attemptRepository.save(attempt);
            eventRepository.save(event);
            outboxPort.enqueueTaskCommand(task, TaskApplicationService.CREATE_TASK_OPERATION, creation.workerPayload());
            idempotencyRepository.save(toIdempotencyRecord(creation, task.getId(), result, now));
            auditWritePort.writeAudit("ALGORITHM_TEST_CREATED", creation.userId(), task.getId(), Map.of(
                    "trace_id", creation.traceId()
            ));
        } catch (DuplicateTaskIdempotencyException ex) {
            TaskIdempotencyRecord reloaded = idempotencyRepository
                    .findByUserIdAndOperationTypeAndIdempotencyKey(
                            creation.userId(),
                            TaskApplicationService.CREATE_TASK_OPERATION,
                            creation.idempotencyKey()
                    )
                    .orElseThrow(() -> new AlgorithmTestException(
                            TASK_COMMAND_WRITE_FAILED_CODE,
                            "任务幂等记录冲突后重读失败",
                            ex
                    ));
            return resultFromIdempotency(creation, reloaded);
        } catch (AlgorithmTestException ex) {
            throw ex;
        } catch (RuntimeException ex) {
            throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "任务命令写入失败", ex);
        }

        return result;
    }

    private AlgorithmTestCreatedResult resultFromIdempotency(
            AlgorithmTestTaskCreation creation,
            TaskIdempotencyRecord record
    ) {
        if (!record.getRequestHash().equals(creation.requestHash())) {
            throw new AlgorithmTestException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        return parseCreatedSnapshot(record.getResponseSnapshot());
    }

    private TaskIdempotencyRecord toIdempotencyRecord(
            AlgorithmTestTaskCreation creation,
            String taskId,
            AlgorithmTestCreatedResult result,
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

    private String toCreatedSnapshot(AlgorithmTestCreatedResult result) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("task_id", result.taskId());
        snapshot.put("task_type", result.taskType());
        snapshot.put("status", result.status());
        snapshot.put("status_label", result.statusLabel());
        snapshot.put("created_at", result.createdAt().toString());
        try {
            return objectMapper.writeValueAsString(snapshot);
        } catch (JsonProcessingException ex) {
            throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照序列化失败", ex);
        }
    }

    private AlgorithmTestCreatedResult parseCreatedSnapshot(String responseSnapshot) {
        try {
            JsonNode root = objectMapper.readTree(responseSnapshot);
            return new AlgorithmTestCreatedResult(
                    requiredText(root, "task_id"),
                    requiredText(root, "task_type"),
                    requiredText(root, "status"),
                    requiredText(root, "status_label"),
                    Instant.parse(requiredText(root, "created_at"))
            );
        } catch (RuntimeException | JsonProcessingException ex) {
            throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照格式非法", ex);
        }
    }

    private Optional<TaskResult> findProblemSet(String taskId) {
        return taskResultRepository.findByTaskId(taskId).stream()
                .filter(result -> result.getResultType() == ResultType.ALGORITHM_TEST)
                .filter(result -> "problem_set".equals(result.getSummary().get("phase")))
                .max(Comparator.comparingInt(TaskResult::getAttemptNo));
    }

    private boolean hasFinalCheckResult(String taskId) {
        return taskResultRepository.findByTaskId(taskId).stream()
                .filter(result -> result.getResultType() == ResultType.ALGORITHM_TEST)
                .anyMatch(result -> "check_result".equals(result.getSummary().get("phase")));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> checkPayload(Task task, TaskResult problemSet) {
        Map<String, Object> inputSummary = new LinkedHashMap<>();
        inputSummary.put("schema_version", 1);
        inputSummary.put("action", "check_submission");
        inputSummary.put("problem_count", AlgorithmTestApplicationService.PROBLEM_COUNT);
        inputSummary.put("pass_threshold", AlgorithmTestApplicationService.PASS_THRESHOLD);

        try {
            JsonNode root = objectMapper.readTree(task.getInputSummary());
            JsonNode targetJob = root.path("target_job");
            JsonNode codeforcesHandle = root.path("codeforces_handle");
            if (!targetJob.isObject()) {
                throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "算法测试任务目标岗位摘要缺失");
            }
            if (!codeforcesHandle.isTextual() || codeforcesHandle.asText().isBlank()) {
                throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "算法测试任务 Codeforces 账号摘要缺失");
            }
            inputSummary.put("target_job", objectMapper.convertValue(targetJob, Map.class));
            inputSummary.put("codeforces_handle", codeforcesHandle.asText());
        } catch (JsonProcessingException ex) {
            throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "算法测试任务输入摘要格式非法", ex);
        }

        inputSummary.put("problem_set_version", problemSet.getSummary().getOrDefault("problem_set_version", 1));
        inputSummary.put("problems", validProblemSetProblems(problemSet));

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("task_type", "algorithm_test");
        payload.put("user_id", task.getUserId());
        payload.put("input_summary", inputSummary);
        payload.put("input_files", List.of());
        return payload;
    }

    private List<Map<String, Object>> validProblemSetProblems(TaskResult problemSet) {
        Object problems = problemSet.getSummary().get("problems");
        if (!(problems instanceof List<?> problemItems)
                || problemItems.size() != AlgorithmTestApplicationService.PROBLEM_COUNT) {
            throw new AlgorithmTestException(PROBLEM_SET_MISSING_CODE, "题单数据不完整");
        }
        List<Map<String, Object>> normalizedProblems = new ArrayList<>();
        for (Object item : problemItems) {
            normalizedProblems.add(validProblemPayload(item));
        }
        return normalizedProblems;
    }

    private Map<String, Object> validProblemPayload(Object item) {
        if (!(item instanceof Map<?, ?> rawProblem)) {
            throw new AlgorithmTestException(PROBLEM_SET_MISSING_CODE, "题单数据不完整");
        }
        Map<String, Object> problem = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : rawProblem.entrySet()) {
            if (entry.getKey() instanceof String key) {
                problem.put(key, entry.getValue());
            }
        }

        Object id = problem.get("id");
        Object contestId = problem.get("contest_id");
        Object problemIndex = problem.get("problem_index");
        Object title = problem.get("title");
        if (!(id instanceof Number)
                || !(contestId instanceof Number contestNumber)
                || !(problemIndex instanceof String problemIndexText)
                || problemIndexText.isBlank()
                || !(title instanceof String titleText)
                || titleText.isBlank()) {
            throw new AlgorithmTestException(PROBLEM_SET_MISSING_CODE, "题单数据不完整");
        }

        problem.put("contest_id", contestNumber.intValue());
        problem.put("problem_index", problemIndexText.trim());
        problem.put("title", titleText.trim());
        return problem;
    }

    private AlgorithmCheckRequestedResult checkResultFromIdempotency(
            AlgorithmTestCheckRequest request,
            TaskIdempotencyRecord record
    ) {
        if (!record.getRequestHash().equals(request.requestHash())) {
            throw new AlgorithmTestException(IDEMPOTENCY_HASH_CONFLICT_CODE, "重复提交参数不一致");
        }
        return parseCheckSnapshot(record.getResponseSnapshot());
    }

    private TaskIdempotencyRecord toCheckIdempotencyRecord(
            AlgorithmTestCheckRequest request,
            String taskId,
            AlgorithmCheckRequestedResult result,
            Instant now
    ) {
        return new TaskIdempotencyRecord(
                idSupplier.get(),
                request.userId(),
                request.idempotencyKey(),
                CHECK_OPERATION,
                request.requestHash(),
                taskId,
                toCheckSnapshot(result),
                now.plusSeconds(86400),
                now,
                now
        );
    }

    private String toCheckSnapshot(AlgorithmCheckRequestedResult result) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("task_id", result.taskId());
        snapshot.put("status", result.status());
        snapshot.put("status_label", result.statusLabel());
        snapshot.put("attempt", result.attempt());
        try {
            return objectMapper.writeValueAsString(snapshot);
        } catch (JsonProcessingException ex) {
            throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照序列化失败", ex);
        }
    }

    private AlgorithmCheckRequestedResult parseCheckSnapshot(String responseSnapshot) {
        try {
            JsonNode root = objectMapper.readTree(responseSnapshot);
            return new AlgorithmCheckRequestedResult(
                    requiredText(root, "task_id"),
                    requiredText(root, "status"),
                    requiredText(root, "status_label"),
                    requiredInt(root, "attempt")
            );
        } catch (RuntimeException | JsonProcessingException ex) {
            throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "幂等响应快照格式非法", ex);
        }
    }

    private static int requiredInt(JsonNode root, String field) {
        JsonNode value = root.get(field);
        if (value == null || !value.canConvertToInt()) {
            throw new IllegalArgumentException("缺少字段 " + field);
        }
        return value.asInt();
    }

    private static String requiredText(JsonNode root, String field) {
        JsonNode value = root.get(field);
        if (value == null || !value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException("缺少字段 " + field);
        }
        return value.asText();
    }
}
