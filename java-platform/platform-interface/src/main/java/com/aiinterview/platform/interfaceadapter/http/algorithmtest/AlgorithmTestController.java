package com.aiinterview.platform.interfaceadapter.http.algorithmtest;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireCsrfToken;
import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.llm.LlmConfig;
import com.aiinterview.platform.application.algorithmtest.AlgorithmTestApplicationService;
import com.aiinterview.platform.application.algorithmtest.AlgorithmTestException;
import com.aiinterview.platform.application.algorithmtest.command.CreateAlgorithmTestCommand;
import com.aiinterview.platform.application.algorithmtest.command.RequestAlgorithmCheckCommand;
import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmCheckRequestedResult;
import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmTestCreatedResult;
import com.aiinterview.platform.application.auth.UserLlmConfigApplicationService;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.http.common.request.LlmConfigRequestMapper;
import com.aiinterview.platform.interfaceadapter.http.algorithmtest.request.CreateAlgorithmTestRequest;
import com.aiinterview.platform.interfaceadapter.http.algorithmtest.request.RequestAlgorithmCheckRequest;
import com.aiinterview.platform.interfaceadapter.http.algorithmtest.response.AlgorithmCheckRequestedResponse;
import com.aiinterview.platform.interfaceadapter.http.algorithmtest.response.AlgorithmTestCreatedResponse;
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
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 算法测试任务接口。
 */
@RestController
@RequestMapping("/api/v1/algorithm-tests")
public class AlgorithmTestController {

    private static final int IDEMPOTENCY_KEY_MISSING_CODE = 76001;
    private static final int TARGET_JOB_TITLE_INVALID_CODE = 76002;
    private static final int TARGET_JOB_DESCRIPTION_INVALID_CODE = 76003;
    private static final int CODEFORCES_HANDLE_INVALID_CODE = 76004;
    private static final Pattern CODEFORCES_HANDLE_PATTERN = Pattern.compile("^[A-Za-z0-9_.-]{1,80}$");

    private final AlgorithmTestApplicationService algorithmTestApplicationService;
    private final SessionManager sessionManager;
    private final UserLlmConfigApplicationService userLlmConfigApplicationService;

    public AlgorithmTestController(
            AlgorithmTestApplicationService algorithmTestApplicationService,
            SessionManager sessionManager,
            UserLlmConfigApplicationService userLlmConfigApplicationService
    ) {
        this.algorithmTestApplicationService = Objects.requireNonNull(
                algorithmTestApplicationService,
                "algorithmTestApplicationService 不能为空"
        );
        this.sessionManager = Objects.requireNonNull(sessionManager, "sessionManager 不能为空");
        this.userLlmConfigApplicationService = Objects.requireNonNull(
                userLlmConfigApplicationService,
                "userLlmConfigApplicationService 不能为空"
        );
    }

    @PostMapping("/{taskId}/checks")
    public ResponseEntity<BaseResult<AlgorithmCheckRequestedResponse>> requestCheck(
            @PathVariable String taskId,
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "Idempotency-Key", required = false) String idempotencyKey,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            @RequestBody(required = false) RequestAlgorithmCheckRequest body,
            HttpServletRequest request
    ) {
        requireCsrfToken(csrfToken);
        requireIdempotencyKey(idempotencyKey);
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        AlgorithmCheckRequestedResult result = algorithmTestApplicationService.requestCheck(
                new RequestAlgorithmCheckCommand(
                        currentSession.getAccountId(),
                        taskId,
                        idempotencyKey,
                        checkRequestHash(taskId, body),
                        traceId(traceId)
                )
        );
        return ResponseEntity.status(HttpStatus.ACCEPTED)
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(AlgorithmCheckRequestedResponse.from(result)));
    }

    @PostMapping
    public ResponseEntity<BaseResult<AlgorithmTestCreatedResponse>> createAlgorithmTest(
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "Idempotency-Key", required = false) String idempotencyKey,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            @RequestBody CreateAlgorithmTestRequest body,
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
        AlgorithmTestCreatedResult result = algorithmTestApplicationService.createAlgorithmTest(
                new CreateAlgorithmTestCommand(
                        currentSession.getAccountId(),
                        normalized.title(),
                        normalized.description(),
                        normalized.codeforcesHandle(),
                        llmConfig,
                        idempotencyKey,
                        requestHash(normalized, llmConfig),
                        traceId(traceId)
                )
        );
        return ResponseEntity.status(HttpStatus.ACCEPTED)
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(AlgorithmTestCreatedResponse.from(result)));
    }

    private static void requireIdempotencyKey(String idempotencyKey) {
        if (idempotencyKey == null || idempotencyKey.isBlank()) {
            throw new AlgorithmTestException(IDEMPOTENCY_KEY_MISSING_CODE, "创建算法测试任务必须提供幂等 key");
        }
    }

    private static NormalizedRequest validateAndNormalize(CreateAlgorithmTestRequest body) {
        if (body == null || body.getTargetJob() == null) {
            throw new AlgorithmTestException(TARGET_JOB_TITLE_INVALID_CODE, "target_job.title 必填");
        }
        String title = trim(body.getTargetJob().getTitle());
        if (title == null || title.isBlank() || title.length() > 100) {
            throw new AlgorithmTestException(TARGET_JOB_TITLE_INVALID_CODE, "target_job.title 必填且长度不能超过 100");
        }
        String description = trim(body.getTargetJob().getDescription());
        if (description == null || description.isBlank() || description.length() < 20 || description.length() > 4000) {
            throw new AlgorithmTestException(TARGET_JOB_DESCRIPTION_INVALID_CODE, "target_job.description 不合法");
        }
        String handle = trim(body.getCodeforcesHandle());
        if (handle == null || handle.isBlank() || !CODEFORCES_HANDLE_PATTERN.matcher(handle).matches()) {
            throw new AlgorithmTestException(CODEFORCES_HANDLE_INVALID_CODE, "codeforces_handle 不合法");
        }
        return new NormalizedRequest(title, description, handle);
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
                    request.title() + ":" + request.description() + ":" + request.codeforcesHandle()
                            + ":" + LlmConfigRequestMapper.publicHashPart(llmConfig)
            ).getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 摘要算法不可用", ex);
        }
    }

    private static String checkRequestHash(String taskId, RequestAlgorithmCheckRequest body) {
        Boolean forceRefresh = body == null ? null : body.getForceRefresh();
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest((taskId + ":" + forceRefresh).getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 摘要算法不可用", ex);
        }
    }

    private record NormalizedRequest(String title, String description, String codeforcesHandle) {
    }
}
