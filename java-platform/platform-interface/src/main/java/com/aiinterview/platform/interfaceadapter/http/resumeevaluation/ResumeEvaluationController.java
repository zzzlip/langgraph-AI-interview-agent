package com.aiinterview.platform.interfaceadapter.http.resumeevaluation;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireCsrfToken;
import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.auth.UserLlmConfigApplicationService;
import com.aiinterview.platform.application.llm.LlmConfig;
import com.aiinterview.platform.application.resumeevaluation.ResumeEvaluationApplicationService;
import com.aiinterview.platform.application.resumeevaluation.command.CreateResumeEvaluationCommand;
import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationCreatedResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.error.ValidationException;
import com.aiinterview.platform.interfaceadapter.http.common.request.LlmConfigRequestMapper;
import com.aiinterview.platform.interfaceadapter.http.resumeevaluation.request.CreateResumeEvaluationRequest;
import com.aiinterview.platform.interfaceadapter.http.resumeevaluation.response.ResumeEvaluationCreatedResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.UUID;
import org.springframework.http.CacheControl;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 简历评估任务接口。
 */
@RestController
@RequestMapping("/api/v1/resume-evaluations")
public class ResumeEvaluationController {

    private final ResumeEvaluationApplicationService resumeEvaluationApplicationService;
    private final SessionManager sessionManager;
    private final UserLlmConfigApplicationService userLlmConfigApplicationService;

    public ResumeEvaluationController(
            ResumeEvaluationApplicationService resumeEvaluationApplicationService,
            SessionManager sessionManager,
            UserLlmConfigApplicationService userLlmConfigApplicationService
    ) {
        this.resumeEvaluationApplicationService = Objects.requireNonNull(
                resumeEvaluationApplicationService,
                "resumeEvaluationApplicationService 不能为空"
        );
        this.sessionManager = Objects.requireNonNull(sessionManager, "sessionManager 不能为空");
        this.userLlmConfigApplicationService = Objects.requireNonNull(
                userLlmConfigApplicationService,
                "userLlmConfigApplicationService 不能为空"
        );
    }

    @PostMapping
    public ResponseEntity<BaseResult<ResumeEvaluationCreatedResponse>> createEvaluation(
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "Idempotency-Key", required = false) String idempotencyKey,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            @Valid @RequestBody CreateResumeEvaluationRequest body,
            HttpServletRequest request
    ) {
        requireCsrfToken(csrfToken);
        requireIdempotencyKey(idempotencyKey);
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        String title = trim(body.getTargetJob().getTitle());
        String description = trim(body.getTargetJob().getDescription());
        LlmConfig llmConfig = LlmConfigRequestMapper.toCommandConfig(
                currentSession.getAccountId(),
                body.getLlmConfig(),
                userLlmConfigApplicationService
        );
        ResumeEvaluationCreatedResult result = resumeEvaluationApplicationService.createEvaluation(
                new CreateResumeEvaluationCommand(
                        currentSession.getAccountId(),
                        body.getResumeFileId(),
                        title,
                        description,
                        llmConfig,
                        idempotencyKey,
                        requestHash(body.getResumeFileId(), title, description, llmConfig),
                        traceId(traceId)
                )
        );
        return ResponseEntity.status(HttpStatus.CREATED)
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(ResumeEvaluationCreatedResponse.from(result)));
    }

    private static void requireIdempotencyKey(String idempotencyKey) {
        if (idempotencyKey == null || idempotencyKey.isBlank()) {
            throw new ValidationException("Idempotency-Key", "不能为空");
        }
    }

    private static String traceId(String traceId) {
        return traceId == null || traceId.isBlank() ? UUID.randomUUID().toString() : traceId;
    }

    private static String trim(String value) {
        return value == null ? null : value.trim();
    }

    private static String requestHash(
            String resumeFileId,
            String targetJobTitle,
            String targetJobDescription,
            LlmConfig llmConfig
    ) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest((
                    resumeFileId + ":" + targetJobTitle + ":" + targetJobDescription + ":"
                            + LlmConfigRequestMapper.publicHashPart(llmConfig)
            ).getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 摘要算法不可用", ex);
        }
    }
}
