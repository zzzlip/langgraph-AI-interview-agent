package com.aiinterview.platform.application.mockinterview;
import com.aiinterview.platform.application.mockinterview.command.WorkerMockInterviewFeedbackCommand;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCurrentState;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewStateSnapshot;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewMemoryCachePort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewResultPort;
import com.aiinterview.platform.application.task.TaskApplicationService;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskEvent;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskEventType;
import com.aiinterview.platform.domain.task.TaskOperatorType;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.aiinterview.platform.domain.task.TaskType;
import java.time.Clock;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;

/**
 * 模拟面试中间反馈处理服务。
 */
public class MockInterviewFeedbackService {

    private static final int TASK_NOT_FOUND_CODE = 71003;
    private static final int ATTEMPT_MISMATCH_CODE = 71011;
    private static final int VERSION_CONFLICT_CODE = 71009;
    private static final int TASK_TYPE_INVALID_CODE = 77007;
    private static final int TASK_STATUS_INVALID_CODE = 77008;
    private static final int QUESTIONS_INVALID_CODE = 77009;
    private static final int FEEDBACK_INVALID_CODE = 77013;
    private static final String QUESTIONS_EVENT_TYPE = "mock_interview.questions.generated";
    private static final String FOLLOW_UP_EVENT_TYPE = "mock_interview.follow_up.generated";
    private static final String QUESTION_COMPLETED_EVENT_TYPE = "mock_interview.question.completed";
    private static final String NEEDS_RESUBMIT_EVENT_TYPE = "mock_interview.answer.needs_resubmit";
    private static final String TASK_TYPE = "mock_interview";
    private static final String WAITING_STATUS = "waiting_user_input";
    private static final Set<String> QUESTION_TYPES = Set.of(
            "technical_fundamental",
            "project_experience",
            "business_scenario",
            "soft_skill"
    );

    private final TaskRepository taskRepository;
    private final TaskEventRepository eventRepository;
    private final MockInterviewResultPort resultPort;
    private final TaskCommandOutboxPort outboxPort;
    private final Clock clock;
    private final Supplier<String> idSupplier;
    private final MockInterviewMemoryCachePort memoryCachePort;

    public MockInterviewFeedbackService(
            TaskRepository taskRepository,
            TaskEventRepository eventRepository,
            MockInterviewResultPort resultPort,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this(taskRepository, eventRepository, resultPort, null, clock, idSupplier, null);
    }

    public MockInterviewFeedbackService(
            TaskRepository taskRepository,
            TaskEventRepository eventRepository,
            MockInterviewResultPort resultPort,
            TaskCommandOutboxPort outboxPort,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this(taskRepository, eventRepository, resultPort, outboxPort, clock, idSupplier, null);
    }

    public MockInterviewFeedbackService(
            TaskRepository taskRepository,
            TaskEventRepository eventRepository,
            MockInterviewResultPort resultPort,
            TaskCommandOutboxPort outboxPort,
            Clock clock,
            Supplier<String> idSupplier,
            MockInterviewMemoryCachePort memoryCachePort
    ) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.eventRepository = Objects.requireNonNull(eventRepository, "eventRepository 不能为空");
        this.resultPort = Objects.requireNonNull(resultPort, "resultPort 不能为空");
        this.outboxPort = outboxPort;
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
        this.memoryCachePort = memoryCachePort;
    }

    @Transactional
    public boolean handleQuestionsGenerated(WorkerMockInterviewFeedbackCommand command) {
        validateEnvelope(command);
        Task task = taskRepository.findById(command.taskId())
                .orElseThrow(() -> new MockInterviewException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (task.getTaskType() != TaskType.MOCK_INTERVIEW) {
            throw new MockInterviewException(TASK_TYPE_INVALID_CODE, "只能处理模拟面试反馈");
        }
        if (command.attemptNo() < task.getCurrentAttempt()) {
            appendIgnored(task, command, "过期模拟面试题目反馈被忽略");
            return true;
        }
        if (command.attemptNo() > task.getCurrentAttempt()) {
            throw new MockInterviewException(ATTEMPT_MISMATCH_CODE, "Worker 反馈 attempt 不匹配");
        }
        if (!task.getTraceId().equals(command.traceId())) {
            throw new MockInterviewException(ATTEMPT_MISMATCH_CODE, "Worker 反馈 trace_id 不匹配");
        }
        if (task.getStatus().isFinal()) {
            appendIgnored(task, command, "终态任务忽略模拟面试题目反馈");
            return true;
        }
        if (task.getStatus() != TaskStatus.RUNNING) {
            throw new MockInterviewException(TASK_STATUS_INVALID_CODE, "任务不在题目生成运行状态");
        }

        PlanData plan = validatePlan(command.payload(), expectedQuestionCount(task));
        TaskStatus fromStatus = task.getStatus();
        long expectedVersion = task.getVersion();
        Instant now = Instant.now(clock);
        String currentRoundId = idSupplier.get();
        Instant expiresAt = now.plusSeconds(72 * 3600L);
        resultPort.saveInitialPlan(new MockInterviewStateSnapshot(
                task.getId(),
                task.getUserId(),
                command.attemptNo(),
                plan.currentQuestionId(),
                currentRoundId,
                "initial_answer",
                now,
                expiresAt,
                statePayload(plan, currentRoundId)
        ));
        refreshMemoryCache(task.getId(), command.attemptNo(), 0L, plan.currentQuestionId(), currentRoundId, plan.memory());
        task.markWaitingUserInput(command.stage(), command.message(), now);
        if (!taskRepository.updateByIdAndVersion(task, expectedVersion)) {
            throw new MockInterviewException(VERSION_CONFLICT_CODE, "任务版本冲突");
        }
        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                command.attemptNo(),
                TaskEventType.TASK_WAITING_USER_INPUT,
                fromStatus,
                task.getStatus(),
                command.stage(),
                command.progress(),
                command.message(),
                eventPayload(command, plan.currentQuestionId(), currentRoundId),
                TaskOperatorType.WORKER,
                "worker",
                command.traceId(),
                now,
                now
        ));
        return false;
    }

    @Transactional
    public boolean handleIntermediateFeedback(WorkerMockInterviewFeedbackCommand command) {
        if (QUESTIONS_EVENT_TYPE.equals(command.eventType())) {
            return handleQuestionsGenerated(command);
        }
        if (!Set.of(FOLLOW_UP_EVENT_TYPE, QUESTION_COMPLETED_EVENT_TYPE, NEEDS_RESUBMIT_EVENT_TYPE)
                .contains(command.eventType())) {
            throw new MockInterviewException(FEEDBACK_INVALID_CODE, "模拟面试中间反馈类型不支持");
        }
        validateIntermediateEnvelope(command);
        Task task = taskRepository.findById(command.taskId())
                .orElseThrow(() -> new MockInterviewException(TASK_NOT_FOUND_CODE, "任务不存在"));
        validateFeedbackTask(command, task);
        MockInterviewCurrentState state = resultPort.findCurrentState(task.getId(), command.attemptNo())
                .orElseThrow(() -> new MockInterviewException(FEEDBACK_INVALID_CODE, "模拟面试状态不存在"));
        TaskStatus fromStatus = task.getStatus();
        long expectedVersion = task.getVersion();
        Instant now = Instant.now(clock);
        StateUpdate update = switch (command.eventType()) {
            case FOLLOW_UP_EVENT_TYPE -> followUpUpdate(state, command);
            case NEEDS_RESUBMIT_EVENT_TYPE -> resubmitUpdate(state, command);
            case QUESTION_COMPLETED_EVENT_TYPE -> questionCompletedUpdate(task, state, command);
            default -> throw new MockInterviewException(FEEDBACK_INVALID_CODE, "模拟面试中间反馈类型不支持");
        };
        if (!resultPort.updateCurrentState(
                state,
                update.currentQuestionId(),
                update.currentRoundId(),
                update.waitingInputType(),
                update.statePayload()
        )) {
            throw new MockInterviewException(VERSION_CONFLICT_CODE, "模拟面试状态版本冲突");
        }
        refreshMemoryCache(
                task.getId(),
                command.attemptNo(),
                state.version() + 1,
                update.currentQuestionId(),
                update.currentRoundId(),
                update.statePayload().get("task_memory")
        );
        if (update.waiting()) {
            task.markWaitingUserInput(command.stage(), command.message(), now);
        }
        if (!taskRepository.updateByIdAndVersion(task, expectedVersion)) {
            throw new MockInterviewException(VERSION_CONFLICT_CODE, "任务版本冲突");
        }
        if (update.finalizePayload() != null && outboxPort != null) {
            outboxPort.enqueueTaskCommand(task, TaskApplicationService.CREATE_TASK_OPERATION, update.finalizePayload());
        }
        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                command.attemptNo(),
                update.waiting() ? TaskEventType.TASK_WAITING_USER_INPUT : TaskEventType.TASK_PROGRESS_UPDATED,
                fromStatus,
                task.getStatus(),
                command.stage(),
                command.progress(),
                command.message(),
                eventPayload(command, update.currentQuestionId(), update.currentRoundId(), update.waitingInputType()),
                TaskOperatorType.WORKER,
                "worker",
                command.traceId(),
                now,
                now
        ));
        return false;
    }

    private static void validateEnvelope(WorkerMockInterviewFeedbackCommand command) {
        if (command == null
                || !QUESTIONS_EVENT_TYPE.equals(command.eventType())
                || !TASK_TYPE.equals(command.taskType())
                || command.attemptNo() < 1
                || !WAITING_STATUS.equals(command.status())
                || !WAITING_STATUS.equals(command.stage())
                || command.message() == null
                || command.message().isBlank()) {
            throw new MockInterviewException(QUESTIONS_INVALID_CODE, "模拟面试题目反馈格式非法");
        }
    }

    private static void validateIntermediateEnvelope(WorkerMockInterviewFeedbackCommand command) {
        if (command == null
                || !TASK_TYPE.equals(command.taskType())
                || command.attemptNo() < 1
                || command.message() == null
                || command.message().isBlank()
                || command.payload() == null) {
            throw new MockInterviewException(FEEDBACK_INVALID_CODE, "模拟面试中间反馈格式非法");
        }
    }

    private void validateFeedbackTask(WorkerMockInterviewFeedbackCommand command, Task task) {
        if (task.getTaskType() != TaskType.MOCK_INTERVIEW) {
            throw new MockInterviewException(TASK_TYPE_INVALID_CODE, "只能处理模拟面试反馈");
        }
        if (command.attemptNo() != task.getCurrentAttempt() || !task.getTraceId().equals(command.traceId())) {
            throw new MockInterviewException(ATTEMPT_MISMATCH_CODE, "Worker 反馈 attempt 或 trace_id 不匹配");
        }
        if (task.getStatus() != TaskStatus.RUNNING) {
            throw new MockInterviewException(TASK_STATUS_INVALID_CODE, "任务不在回答处理运行状态");
        }
    }

    private StateUpdate followUpUpdate(MockInterviewCurrentState state, WorkerMockInterviewFeedbackCommand command) {
        String followUpQuestion = textPayload(command.payload(), "follow_up_question");
        if (followUpQuestion == null || followUpQuestion.contains("参考答案") || followUpQuestion.contains("标准答案")) {
            throw new MockInterviewException(FEEDBACK_INVALID_CODE, "追问内容非法");
        }
        Map<String, Object> payload = mutablePayload(state);
        String nextRoundId = idSupplier.get();
        List<Object> rounds = rounds(payload);
        rounds.add(new LinkedHashMap<>(Map.of(
                "round_id", nextRoundId,
                "question_id", state.currentQuestionId(),
                "round_type", "follow_up_answer",
                "round_index", rounds.size() + 1,
                "follow_up_question", followUpQuestion,
                "status", "waiting_answer"
        )));
        payload.put("rounds", rounds);
        payload.put("task_memory", command.payload().getOrDefault("task_memory", payload.get("task_memory")));
        payload.put("current_question_id", state.currentQuestionId());
        payload.put("current_round_id", nextRoundId);
        payload.put("waiting_input_type", "follow_up_answer");
        return new StateUpdate(state.currentQuestionId(), nextRoundId, "follow_up_answer", payload, true, null);
    }

    private StateUpdate resubmitUpdate(MockInterviewCurrentState state, WorkerMockInterviewFeedbackCommand command) {
        Map<String, Object> payload = mutablePayload(state);
        payload.put("waiting_input_type", "resubmit_answer");
        payload.put("resubmit_reason", command.message());
        return new StateUpdate(state.currentQuestionId(), state.currentRoundId(), "resubmit_answer", payload, true, null);
    }

    private StateUpdate questionCompletedUpdate(Task task, MockInterviewCurrentState state, WorkerMockInterviewFeedbackCommand command) {
        Map<String, Object> payload = mutablePayload(state);
        String nextQuestionId = nextQuestionId(payload, state.currentQuestionId());
        if (nextQuestionId == null) {
            payload.put("waiting_input_type", "finalizing");
            return new StateUpdate(
                    state.currentQuestionId(),
                    state.currentRoundId(),
                    "finalizing",
                    payload,
                    false,
                    finalizePayload(task, payload, command.traceId())
            );
        }
        String nextRoundId = idSupplier.get();
        List<Object> rounds = rounds(payload);
        rounds.add(new LinkedHashMap<>(Map.of(
                "round_id", nextRoundId,
                "question_id", nextQuestionId,
                "round_type", "initial_answer",
                "round_index", 1,
                "status", "waiting_answer"
        )));
        payload.put("rounds", rounds);
        payload.put("current_question_id", nextQuestionId);
        payload.put("current_round_id", nextRoundId);
        payload.put("waiting_input_type", "initial_answer");
        return new StateUpdate(nextQuestionId, nextRoundId, "initial_answer", payload, true, null);
    }

    private Map<String, Object> finalizePayload(Task task, Map<String, Object> statePayload, String traceId) {
        Map<String, Object> inputSummary = new LinkedHashMap<>();
        inputSummary.put("schema_version", 1);
        inputSummary.put("action", "finalize_analysis");
        inputSummary.put("task_id", task.getId());
        inputSummary.put("attempt", task.getCurrentAttempt());
        inputSummary.put("trace_id", traceId);
        inputSummary.put("interview_state", statePayload);
        Object llmConfig = statePayload.get("llm_config");
        if (llmConfig != null) {
            inputSummary.put("llm_config", llmConfig);
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("task_type", TASK_TYPE);
        payload.put("user_id", task.getUserId());
        payload.put("input_summary", inputSummary);
        payload.put("input_files", List.of());
        return payload;
    }

    private static int expectedQuestionCount(Task task) {
        String inputSummary = task.getInputSummary();
        if (inputSummary != null && inputSummary.contains("\"question_count\":8")) {
            return 8;
        }
        if (inputSummary != null && inputSummary.contains("\"question_count\":4")) {
            return 4;
        }
        return 6;
    }

    private static PlanData validatePlan(Map<String, Object> payload, int expectedQuestionCount) {
        if (payload == null
                || !(payload.get("schema_version") instanceof Integer)
                || !(payload.get("questions") instanceof List<?> rawQuestions)
                || rawQuestions.size() != expectedQuestionCount
                || !(payload.get("current_question_id") instanceof String currentQuestionId)
                || currentQuestionId.isBlank()) {
            throw new MockInterviewException(QUESTIONS_INVALID_CODE, "模拟面试题目数据不完整");
        }
        List<Map<String, Object>> questions = new ArrayList<>();
        Set<Integer> sequences = new HashSet<>();
        Set<String> questionIds = new HashSet<>();
        for (Object rawQuestion : rawQuestions) {
            if (!(rawQuestion instanceof Map<?, ?> question)) {
                throw new MockInterviewException(QUESTIONS_INVALID_CODE, "模拟面试题目格式非法");
            }
            String questionId = requiredText(question, "question_id");
            int sequence = requiredNumber(question, "sequence");
            String questionType = requiredText(question, "question_type");
            String title = requiredText(question, "title");
            String source = requiredText(question, "source");
            String status = requiredText(question, "status");
            if (!QUESTION_TYPES.contains(questionType)
                    || !sequences.add(sequence)
                    || !questionIds.add(questionId)
                    || !"waiting_answer".equals(status)) {
                throw new MockInterviewException(QUESTIONS_INVALID_CODE, "模拟面试题目 schema 非法");
            }
            Map<String, Object> normalized = new LinkedHashMap<>();
            normalized.put("question_id", questionId);
            normalized.put("sequence", sequence);
            normalized.put("question_type", questionType);
            normalized.put("title", title);
            normalized.put("source", source);
            normalized.put("status", status);
            questions.add(normalized);
        }
        if (!questionIds.contains(currentQuestionId)) {
            throw new MockInterviewException(QUESTIONS_INVALID_CODE, "当前题目不在题单内");
        }
        Object memory = payload.get("memory") instanceof Map<?, ?> memoryMap
                ? new LinkedHashMap<>(memoryMap)
                : defaultMemory();
        Object llmConfig = payload.get("llm_config") instanceof Map<?, ?> configMap
                ? new LinkedHashMap<>(configMap)
                : null;
        Object publicLlmConfig = payload.get("public_llm_config") instanceof Map<?, ?> publicConfigMap
                ? new LinkedHashMap<>(publicConfigMap)
                : null;
        return new PlanData(questions, currentQuestionId, memory, llmConfig, publicLlmConfig);
    }

    private static String requiredText(Map<?, ?> map, String field) {
        Object value = map.get(field);
        if (!(value instanceof String text) || text.isBlank()) {
            throw new MockInterviewException(QUESTIONS_INVALID_CODE, field + " 不完整");
        }
        return text;
    }

    private static int requiredNumber(Map<?, ?> map, String field) {
        Object value = map.get(field);
        if (!(value instanceof Number number)) {
            throw new MockInterviewException(QUESTIONS_INVALID_CODE, field + " 不完整");
        }
        return number.intValue();
    }

    private static Map<String, Object> statePayload(PlanData plan, String currentRoundId) {
        Map<String, Object> interviewPlan = new LinkedHashMap<>();
        interviewPlan.put("questions", plan.questions());

        Map<String, Object> round = new LinkedHashMap<>();
        round.put("round_id", currentRoundId);
        round.put("question_id", plan.currentQuestionId());
        round.put("round_type", "initial_answer");
        round.put("round_index", 1);
        round.put("status", "waiting_answer");

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("interview_plan", interviewPlan);
        payload.put("rounds", List.of(round));
        payload.put("task_memory", plan.memory());
        if (plan.llmConfig() != null) {
            payload.put("llm_config", plan.llmConfig());
        }
        if (plan.publicLlmConfig() != null) {
            payload.put("public_llm_config", plan.publicLlmConfig());
        }
        payload.put("current_question_id", plan.currentQuestionId());
        payload.put("current_round_id", currentRoundId);
        payload.put("waiting_input_type", "initial_answer");
        payload.put("resubmit_records", List.of());
        payload.put("stage_failures", List.of());
        return payload;
    }

    private static Map<String, Object> eventPayload(
            WorkerMockInterviewFeedbackCommand command,
            String currentQuestionId,
            String currentRoundId
    ) {
        return eventPayload(command, currentQuestionId, currentRoundId, "initial_answer");
    }

    private static Map<String, Object> eventPayload(
            WorkerMockInterviewFeedbackCommand command,
            String currentQuestionId,
            String currentRoundId,
            String waitingInputType
    ) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("mock_interview_event_type", command.eventType());
        payload.put("task_id", command.taskId());
        payload.put("status", WAITING_STATUS);
        payload.put("stage", command.stage());
        payload.put("attempt", command.attemptNo());
        payload.put("trace_id", command.traceId());
        payload.put("current_question_id", currentQuestionId);
        payload.put("current_round_id", currentRoundId);
        payload.put("waiting_input_type", waitingInputType);
        return payload;
    }

    private static Map<String, Object> mutablePayload(MockInterviewCurrentState state) {
        Map<String, Object> payload = new LinkedHashMap<>(state.statePayload());
        payload.putIfAbsent("rounds", List.of());
        payload.putIfAbsent("stage_failures", List.of());
        payload.putIfAbsent("resubmit_records", List.of());
        return payload;
    }

    private static List<Object> rounds(Map<String, Object> payload) {
        List<Object> rounds = new ArrayList<>();
        Object rawRounds = payload.get("rounds");
        if (rawRounds instanceof List<?> list) {
            rounds.addAll(list);
        }
        return rounds;
    }

    private static String nextQuestionId(Map<String, Object> payload, String currentQuestionId) {
        Object interviewPlan = payload.get("interview_plan");
        if (!(interviewPlan instanceof Map<?, ?> plan) || !(plan.get("questions") instanceof List<?> questions)) {
            return null;
        }
        for (int i = 0; i < questions.size(); i++) {
            Object item = questions.get(i);
            if (item instanceof Map<?, ?> question && currentQuestionId.equals(question.get("question_id"))) {
                if (i + 1 < questions.size() && questions.get(i + 1) instanceof Map<?, ?> next) {
                    Object nextId = next.get("question_id");
                    return nextId instanceof String text && !text.isBlank() ? text : null;
                }
                return null;
            }
        }
        return null;
    }

    private static String textPayload(Map<String, Object> payload, String field) {
        Object value = payload.get(field);
        return value instanceof String text && !text.isBlank() ? text : null;
    }

    private static Map<String, Object> defaultMemory() {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("strengths", List.of());
        summary.put("risks", List.of());
        summary.put("communication", "尚未形成表现摘要");

        Map<String, Object> memory = new LinkedHashMap<>();
        memory.put("schema_version", 1);
        memory.put("question_memory", List.of());
        memory.put("candidate_signal_summary", summary);
        return memory;
    }

    private void appendIgnored(Task task, WorkerMockInterviewFeedbackCommand command, String message) {
        Instant now = Instant.now(clock);
        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                command.attemptNo(),
                TaskEventType.TASK_FEEDBACK_IGNORED,
                task.getStatus(),
                task.getStatus(),
                task.getStage(),
                task.getProgress(),
                message,
                Map.of("schema_version", 1),
                TaskOperatorType.WORKER,
                "worker",
                command.traceId(),
                now,
                now
        ));
    }

    private void refreshMemoryCache(
            String taskId,
            int attemptNo,
            long stateVersion,
            String currentQuestionId,
            String currentRoundId,
            Object taskMemory
    ) {
        if (memoryCachePort == null || taskMemory == null) {
            return;
        }
        try {
            memoryCachePort.refresh(taskId, attemptNo, stateVersion, currentQuestionId, currentRoundId, taskMemory);
        } catch (RuntimeException ignored) {
            // Redis 只是热缓存，MySQL mock_interview_state 仍是事实源。
        }
    }

    private record PlanData(
            List<Map<String, Object>> questions,
            String currentQuestionId,
            Object memory,
            Object llmConfig,
            Object publicLlmConfig
    ) {
    }

    private record StateUpdate(
            String currentQuestionId,
            String currentRoundId,
            String waitingInputType,
            Map<String, Object> statePayload,
            boolean waiting,
            Map<String, Object> finalizePayload
    ) {
    }
}
