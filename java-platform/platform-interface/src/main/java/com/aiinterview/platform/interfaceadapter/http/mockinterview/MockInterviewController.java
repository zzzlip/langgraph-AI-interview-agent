package com.aiinterview.platform.interfaceadapter.http.mockinterview;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireCsrfToken;
import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.auth.UserLlmConfigApplicationService;
import com.aiinterview.platform.application.llm.LlmConfig;
import com.aiinterview.platform.application.mockinterview.MockInterviewApplicationService;
import com.aiinterview.platform.application.mockinterview.MockInterviewException;
import com.aiinterview.platform.application.mockinterview.command.CreateMockInterviewCommand;
import com.aiinterview.platform.application.mockinterview.command.SubmitMockInterviewAnswerCommand;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerSubmittedResult;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCurrentState;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCreatedResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.http.common.request.LlmConfigRequestMapper;
import com.aiinterview.platform.interfaceadapter.http.mockinterview.request.CreateMockInterviewRequest;
import com.aiinterview.platform.interfaceadapter.http.mockinterview.request.SubmitMockInterviewAnswerRequest;
import com.aiinterview.platform.interfaceadapter.http.mockinterview.response.MockInterviewAnswerSubmittedResponse;
import com.aiinterview.platform.interfaceadapter.http.mockinterview.response.MockInterviewCreatedResponse;
import com.aiinterview.platform.interfaceadapter.http.mockinterview.response.MockInterviewStateResponse;
import jakarta.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.UUID;
import java.util.regex.Pattern;
import org.springframework.http.CacheControl;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 模拟面试任务接口。
 */
@RestController
@RequestMapping("/api/v1/mock-interviews")
public class MockInterviewController {

    private static final int IDEMPOTENCY_KEY_MISSING_CODE = 77001;
    private static final int RESUME_FILE_ID_INVALID_CODE = 77002;
    private static final int TARGET_JOB_TITLE_INVALID_CODE = 77003;
    private static final int TARGET_JOB_DESCRIPTION_INVALID_CODE = 77004;
    private static final int QUESTION_COUNT_INVALID_CODE = 77005;
    private static final int FOLLOW_UP_ROUNDS_INVALID_CODE = 77006;
    private static final int ROUND_ID_INVALID_CODE = 77011;
    private static final int ANSWER_FILE_ID_INVALID_CODE = 77012;
    private static final int DEFAULT_QUESTION_COUNT = 6;
    private static final int DEFAULT_MAX_FOLLOW_UP_ROUNDS = 2;
    private static final Pattern FILE_ID_PATTERN = Pattern.compile("^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$");

    private final MockInterviewApplicationService mockInterviewApplicationService;
    private final SessionManager sessionManager;
    private final UserLlmConfigApplicationService userLlmConfigApplicationService;

    public MockInterviewController(
            MockInterviewApplicationService mockInterviewApplicationService,
            SessionManager sessionManager,
            UserLlmConfigApplicationService userLlmConfigApplicationService
    ) {
        this.mockInterviewApplicationService = Objects.requireNonNull(
                mockInterviewApplicationService,
                "mockInterviewApplicationService 不能为空"
        );
        this.sessionManager = Objects.requireNonNull(sessionManager, "sessionManager 不能为空");
        this.userLlmConfigApplicationService = Objects.requireNonNull(
                userLlmConfigApplicationService,
                "userLlmConfigApplicationService 不能为空"
        );
    }

    @PostMapping
    public ResponseEntity<BaseResult<MockInterviewCreatedResponse>> createMockInterview(
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "Idempotency-Key", required = false) String idempotencyKey,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            @RequestBody CreateMockInterviewRequest body,
            HttpServletRequest request
    ) {
        requireCsrfToken(csrfToken);
        requireIdempotencyKey(idempotencyKey);
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        NormalizedRequest normalized = validateAndNormalize(body);
        LlmConfig llmConfig = LlmConfigRequestMapper.toCommandConfig(
                currentSession.getAccountId(),
                body.getLlmConfig(),
                userLlmConfigApplicationService
        );
        MockInterviewCreatedResult result = mockInterviewApplicationService.createMockInterview(
                new CreateMockInterviewCommand(
                        currentSession.getAccountId(),
                        normalized.resumeFileId(),
                        normalized.title(),
                        normalized.description(),
                        normalized.questionCount(),
                        normalized.maxFollowUpRounds(),
                        llmConfig,
                        idempotencyKey,
                        requestHash(normalized, llmConfig),
                        traceId(traceId)
                )
        );
        return ResponseEntity.status(HttpStatus.ACCEPTED)
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(MockInterviewCreatedResponse.from(result)));
    }

    @PostMapping("/{taskId}/answers")
    public ResponseEntity<BaseResult<MockInterviewAnswerSubmittedResponse>> submitAnswer(
            @PathVariable String taskId,
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "Idempotency-Key", required = false) String idempotencyKey,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            @RequestBody SubmitMockInterviewAnswerRequest body,
            HttpServletRequest request
    ) {
        requireCsrfToken(csrfToken);
        requireIdempotencyKey(idempotencyKey);
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        NormalizedAnswerRequest normalized = validateAndNormalizeAnswer(body);
        MockInterviewAnswerSubmittedResult result = mockInterviewApplicationService.submitAnswer(
                new SubmitMockInterviewAnswerCommand(
                        currentSession.getAccountId(),
                        taskId,
                        normalized.questionId(),
                        normalized.roundId(),
                        normalized.answerFileId(),
                        idempotencyKey,
                        answerRequestHash(taskId, normalized),
                        traceId(traceId)
                )
        );
        return ResponseEntity.status(HttpStatus.ACCEPTED)
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(MockInterviewAnswerSubmittedResponse.from(result)));
    }

    @GetMapping("/{taskId}/state")
    public ResponseEntity<BaseResult<MockInterviewStateResponse>> getCurrentState(
            @PathVariable String taskId,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        MockInterviewCurrentState state = mockInterviewApplicationService.getCurrentState(
                currentSession.getAccountId(),
                taskId
        );
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(MockInterviewStateResponse.from(state)));
    }

    private static void requireIdempotencyKey(String idempotencyKey) {
        if (idempotencyKey == null || idempotencyKey.isBlank()) {
            throw new MockInterviewException(IDEMPOTENCY_KEY_MISSING_CODE, "创建模拟面试任务必须提供幂等 key");
        }
    }

    private static NormalizedRequest validateAndNormalize(CreateMockInterviewRequest body) {
        if (body == null) {
            throw new MockInterviewException(RESUME_FILE_ID_INVALID_CODE, "resume_file_id 必填");
        }
        String resumeFileId = trim(body.getResumeFileId());
        if (resumeFileId == null || resumeFileId.isBlank() || !FILE_ID_PATTERN.matcher(resumeFileId).matches()) {
            throw new MockInterviewException(RESUME_FILE_ID_INVALID_CODE, "resume_file_id 不合法");
        }
        if (body.getTargetJob() == null) {
            throw new MockInterviewException(TARGET_JOB_TITLE_INVALID_CODE, "target_job.title 必填");
        }
        String title = trim(body.getTargetJob().getTitle());
        if (title == null || title.isBlank() || title.length() > 100) {
            throw new MockInterviewException(TARGET_JOB_TITLE_INVALID_CODE, "target_job.title 必填且长度不能超过 100");
        }
        String description = trim(body.getTargetJob().getDescription());
        if (description == null || description.isBlank() || description.length() < 20 || description.length() > 4000) {
            throw new MockInterviewException(TARGET_JOB_DESCRIPTION_INVALID_CODE, "target_job.description 不合法");
        }
        int questionCount = body.getQuestionCount() == null ? DEFAULT_QUESTION_COUNT : body.getQuestionCount();
        if (questionCount < 4 || questionCount > 8) {
            throw new MockInterviewException(QUESTION_COUNT_INVALID_CODE, "question_count 必须在 4 到 8 之间");
        }
        int maxFollowUpRounds = body.getMaxFollowUpRounds() == null
                ? DEFAULT_MAX_FOLLOW_UP_ROUNDS
                : body.getMaxFollowUpRounds();
        if (maxFollowUpRounds < 1 || maxFollowUpRounds > 3) {
            throw new MockInterviewException(FOLLOW_UP_ROUNDS_INVALID_CODE, "max_follow_up_rounds 必须在 1 到 3 之间");
        }
        return new NormalizedRequest(resumeFileId, title, description, questionCount, maxFollowUpRounds);
    }

    private static NormalizedAnswerRequest validateAndNormalizeAnswer(SubmitMockInterviewAnswerRequest body) {
        if (body == null) {
            throw new MockInterviewException(ROUND_ID_INVALID_CODE, "question_id 和 round_id 必填");
        }
        String questionId = trim(body.getQuestionId());
        String roundId = trim(body.getRoundId());
        String answerFileId = trim(body.getAnswerFileId());
        if (questionId == null || questionId.isBlank() || !FILE_ID_PATTERN.matcher(questionId).matches()) {
            throw new MockInterviewException(ROUND_ID_INVALID_CODE, "question_id 不合法");
        }
        if (roundId == null || roundId.isBlank() || !FILE_ID_PATTERN.matcher(roundId).matches()) {
            throw new MockInterviewException(ROUND_ID_INVALID_CODE, "round_id 不合法");
        }
        if (answerFileId == null || answerFileId.isBlank() || !FILE_ID_PATTERN.matcher(answerFileId).matches()) {
            throw new MockInterviewException(ANSWER_FILE_ID_INVALID_CODE, "answer_file_id 不合法");
        }
        return new NormalizedAnswerRequest(questionId, roundId, answerFileId);
    }

    private static String traceId(String traceId) {
        return traceId == null || traceId.isBlank() ? UUID.randomUUID().toString() : traceId;
    }

    private static String trim(String value) {
        return value == null ? null : value.trim();
    }

    private static String requestHash(NormalizedRequest request, LlmConfig llmConfig) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest((
                    request.resumeFileId()
                            + ":"
                            + request.title()
                            + ":"
                            + request.description()
                            + ":"
                            + request.questionCount()
                            + ":"
                            + request.maxFollowUpRounds()
                            + ":"
                            + LlmConfigRequestMapper.publicHashPart(llmConfig)
            ).getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 摘要算法不可用", ex);
        }
    }

    private static String answerRequestHash(String taskId, NormalizedAnswerRequest request) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest((
                    taskId
                            + ":"
                            + request.questionId()
                            + ":"
                            + request.roundId()
                            + ":"
                            + request.answerFileId()
            ).getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 摘要算法不可用", ex);
        }
    }

    private record NormalizedRequest(
            String resumeFileId,
            String title,
            String description,
            int questionCount,
            int maxFollowUpRounds
    ) {
    }

    private record NormalizedAnswerRequest(
            String questionId,
            String roundId,
            String answerFileId
    ) {
    }
}
