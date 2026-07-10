package com.aiinterview.platform.infrastructure.asyncmessage;

import com.aiinterview.platform.application.algorithmtest.AlgorithmTestProblemSetFeedbackService;
import com.aiinterview.platform.application.algorithmtest.command.AlgorithmProblemSetFeedbackCommand;
import com.aiinterview.platform.application.asyncmessage.AsyncMessageException;
import com.aiinterview.platform.application.asyncmessage.WorkerFeedbackConsumeService;
import com.aiinterview.platform.application.asyncmessage.WorkerFeedbackInboxService;
import com.aiinterview.platform.application.asyncmessage.command.WorkerFeedbackEventCommand;
import com.aiinterview.platform.application.asyncmessage.dto.WorkerFeedbackInboxResult;
import com.aiinterview.platform.application.asyncmessage.port.SseTaskEventPublisher;
import com.aiinterview.platform.application.mockinterview.MockInterviewFeedbackService;
import com.aiinterview.platform.application.mockinterview.command.WorkerMockInterviewFeedbackCommand;
import com.aiinterview.platform.domain.asyncmessage.InboxEvent;
import com.aiinterview.platform.domain.asyncmessage.InboxEventRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * 算法测试题单中间反馈分发器。
 */
public class AlgorithmProblemSetFeedbackDispatcher {

    private static final int WORKER_EVENT_FORMAT_INVALID_CODE = 74005;
    private static final int WORKER_FEEDBACK_HANDLE_FAILED_CODE = 74009;
    private static final String PROBLEM_SET_EVENT_TYPE = "algorithm_test.problem_set.generated";
    private static final String MOCK_INTERVIEW_QUESTIONS_EVENT_TYPE = "mock_interview.questions.generated";
    private static final String MOCK_INTERVIEW_FOLLOW_UP_EVENT_TYPE = "mock_interview.follow_up.generated";
    private static final String MOCK_INTERVIEW_QUESTION_COMPLETED_EVENT_TYPE = "mock_interview.question.completed";
    private static final String MOCK_INTERVIEW_NEEDS_RESUBMIT_EVENT_TYPE = "mock_interview.answer.needs_resubmit";

    private final WorkerFeedbackConsumeService commonConsumeService;
    private final WorkerFeedbackInboxService inboxService;
    private final InboxEventRepository inboxEventRepository;
    private final AlgorithmTestProblemSetFeedbackService problemSetFeedbackService;
    private final MockInterviewFeedbackService mockInterviewFeedbackService;
    private final SseTaskEventPublisher sseTaskEventPublisher;
    private final ObjectMapper objectMapper;
    private final Clock clock;

    public AlgorithmProblemSetFeedbackDispatcher(
            WorkerFeedbackConsumeService commonConsumeService,
            WorkerFeedbackInboxService inboxService,
            InboxEventRepository inboxEventRepository,
            AlgorithmTestProblemSetFeedbackService problemSetFeedbackService,
            SseTaskEventPublisher sseTaskEventPublisher,
            ObjectMapper objectMapper,
            Clock clock
    ) {
        this(commonConsumeService, inboxService, inboxEventRepository, problemSetFeedbackService, null,
                sseTaskEventPublisher, objectMapper, clock);
    }

    public AlgorithmProblemSetFeedbackDispatcher(
            WorkerFeedbackConsumeService commonConsumeService,
            WorkerFeedbackInboxService inboxService,
            InboxEventRepository inboxEventRepository,
            AlgorithmTestProblemSetFeedbackService problemSetFeedbackService,
            MockInterviewFeedbackService mockInterviewFeedbackService,
            SseTaskEventPublisher sseTaskEventPublisher,
            ObjectMapper objectMapper,
            Clock clock
    ) {
        this.commonConsumeService = Objects.requireNonNull(commonConsumeService, "commonConsumeService 不能为空");
        this.inboxService = Objects.requireNonNull(inboxService, "inboxService 不能为空");
        this.inboxEventRepository = Objects.requireNonNull(inboxEventRepository, "inboxEventRepository 不能为空");
        this.problemSetFeedbackService = Objects.requireNonNull(problemSetFeedbackService, "problemSetFeedbackService 不能为空");
        this.mockInterviewFeedbackService = mockInterviewFeedbackService;
        this.sseTaskEventPublisher = Objects.requireNonNull(sseTaskEventPublisher, "sseTaskEventPublisher 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
    }

    public WorkerFeedbackInboxResult dispatch(WorkerFeedbackEventCommand command) {
        if (!PROBLEM_SET_EVENT_TYPE.equals(command.eventType())) {
            if (isMockInterviewIntermediateEvent(command.eventType())) {
                return dispatchMockInterviewIntermediate(command);
            }
            return commonConsumeService.consume(command);
        }

        WorkerFeedbackInboxResult inboxResult = inboxService.recordReceived(command);
        if (inboxResult.status() == WorkerFeedbackInboxResult.Status.DUPLICATE_ACK) {
            return inboxResult;
        }

        InboxEvent processing = inboxResult.event().markProcessing(now());
        inboxEventRepository.update(processing);
        try {
            boolean ignored = problemSetFeedbackService.handleProblemSet(toProblemSetCommand(command));
            InboxEvent completed = ignored
                    ? processing.markIgnored("算法题单反馈被忽略", now())
                    : processing.markProcessed(now());
            inboxEventRepository.update(completed);
            sseTaskEventPublisher.publishTaskChanged(processing.getTaskId());
            return WorkerFeedbackInboxResult.received(completed);
        } catch (RuntimeException ex) {
            InboxEvent failed = processing.markFailed("algorithm problem set feedback failed", now());
            inboxEventRepository.update(failed);
            throw new AsyncMessageException(WORKER_FEEDBACK_HANDLE_FAILED_CODE, "算法题单反馈处理失败", ex);
        }
    }

    private WorkerFeedbackInboxResult dispatchMockInterviewIntermediate(WorkerFeedbackEventCommand command) {
        if (mockInterviewFeedbackService == null) {
            return commonConsumeService.consume(command);
        }
        WorkerFeedbackInboxResult inboxResult = inboxService.recordReceived(command);
        if (inboxResult.status() == WorkerFeedbackInboxResult.Status.DUPLICATE_ACK) {
            return inboxResult;
        }

        InboxEvent processing = inboxResult.event().markProcessing(now());
        inboxEventRepository.update(processing);
        try {
            boolean ignored = mockInterviewFeedbackService.handleIntermediateFeedback(toMockInterviewCommand(command));
            InboxEvent completed = ignored
                    ? processing.markIgnored("模拟面试题目反馈被忽略", now())
                    : processing.markProcessed(now());
            inboxEventRepository.update(completed);
            sseTaskEventPublisher.publishTaskChanged(processing.getTaskId());
            return WorkerFeedbackInboxResult.received(completed);
        } catch (RuntimeException ex) {
            InboxEvent failed = processing.markFailed("mock interview questions feedback failed", now());
            inboxEventRepository.update(failed);
            throw new AsyncMessageException(WORKER_FEEDBACK_HANDLE_FAILED_CODE, "模拟面试题目反馈处理失败", ex);
        }
    }

    private static boolean isMockInterviewIntermediateEvent(String eventType) {
        return MOCK_INTERVIEW_QUESTIONS_EVENT_TYPE.equals(eventType)
                || MOCK_INTERVIEW_FOLLOW_UP_EVENT_TYPE.equals(eventType)
                || MOCK_INTERVIEW_QUESTION_COMPLETED_EVENT_TYPE.equals(eventType)
                || MOCK_INTERVIEW_NEEDS_RESUBMIT_EVENT_TYPE.equals(eventType);
    }

    @SuppressWarnings("unchecked")
    private AlgorithmProblemSetFeedbackCommand toProblemSetCommand(WorkerFeedbackEventCommand command) {
        try {
            JsonNode payload = objectMapper.readTree(command.rawPayload()).path("payload");
            Map<String, Object> resultSummary = objectMapper.convertValue(payload.path("result_summary"), Map.class);
            return new AlgorithmProblemSetFeedbackCommand(
                    command.eventType(),
                    command.taskId(),
                    command.taskType(),
                    command.attemptNo(),
                    command.traceId(),
                    text(payload, "status"),
                    text(payload, "stage"),
                    payload.path("progress").isInt() ? payload.path("progress").asInt() : null,
                    text(payload, "message"),
                    resultSummary
            );
        } catch (Exception ex) {
            throw new AsyncMessageException(WORKER_EVENT_FORMAT_INVALID_CODE, "Worker 事件格式非法", ex);
        }
    }

    @SuppressWarnings("unchecked")
    private WorkerMockInterviewFeedbackCommand toMockInterviewCommand(WorkerFeedbackEventCommand command) {
        try {
            JsonNode payload = objectMapper.readTree(command.rawPayload()).path("payload");
            return new WorkerMockInterviewFeedbackCommand(
                    command.eventType(),
                    command.taskId(),
                    command.taskType(),
                    command.attemptNo(),
                    command.traceId(),
                    text(payload, "status"),
                    text(payload, "stage"),
                    payload.path("progress").isInt() ? payload.path("progress").asInt() : null,
                    text(payload, "message"),
                    objectMapper.convertValue(payload, Map.class)
            );
        } catch (Exception ex) {
            throw new AsyncMessageException(WORKER_EVENT_FORMAT_INVALID_CODE, "Worker 事件格式非法", ex);
        }
    }

    private static String text(JsonNode node, String field) {
        JsonNode value = node.path(field);
        return value.isTextual() ? value.asText() : null;
    }

    private Instant now() {
        return Instant.now(clock);
    }
}
