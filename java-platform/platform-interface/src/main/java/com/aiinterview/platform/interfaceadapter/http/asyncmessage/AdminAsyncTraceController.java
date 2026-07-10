package com.aiinterview.platform.interfaceadapter.http.asyncmessage;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireAdminSession;
import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireCsrfToken;

import com.aiinterview.platform.application.asyncmessage.AsyncAdminQueryService;
import com.aiinterview.platform.application.asyncmessage.dto.AsyncTaskTraceResult;
import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterHandledResult;
import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterPageResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterSourceType;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.error.ValidationException;
import com.aiinterview.platform.interfaceadapter.http.asyncmessage.response.AsyncTaskTraceResponse;
import com.aiinterview.platform.interfaceadapter.http.asyncmessage.response.DeadLetterResponse;
import jakarta.servlet.http.HttpServletRequest;
import java.util.Arrays;
import org.springframework.http.CacheControl;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 管理员异步协作排查接口。
 */
@RestController
@RequestMapping("/api/v1/admin")
public class AdminAsyncTraceController {

    private final AsyncAdminQueryService queryService;
    private final SessionManager sessionManager;

    public AdminAsyncTraceController(AsyncAdminQueryService queryService, SessionManager sessionManager) {
        this.queryService = queryService;
        this.sessionManager = sessionManager;
    }

    @GetMapping("/tasks/{taskId}/async-trace")
    public ResponseEntity<BaseResult<AsyncTaskTraceResponse>> getTaskTrace(@PathVariable String taskId,
            HttpServletRequest request) {
        CurrentSession currentSession = requireAdminSession(request, sessionManager);
        AsyncTaskTraceResult result = queryService.getTaskTrace(currentSession.getAccountId(), taskId);
        return ok(AsyncTaskTraceResponse.from(result));
    }

    @GetMapping("/async-dead-letters")
    public ResponseEntity<BaseResult<DeadLetterResponse.Page>> listDeadLetters(
            @RequestParam(name = "page", defaultValue = "1") int page,
            @RequestParam(name = "page_size", defaultValue = "20") int pageSize,
            @RequestParam(name = "source_type", required = false) String sourceType,
            @RequestParam(name = "handled", required = false) Boolean handled,
            HttpServletRequest request) {
        CurrentSession currentSession = requireAdminSession(request, sessionManager);
        validatePage(page, pageSize);
        DeadLetterPageResult result = queryService.listDeadLetters(currentSession.getAccountId(),
                parseSourceType(sourceType), handled, page, pageSize);
        return ok(DeadLetterResponse.Page.from(result));
    }

    @PostMapping("/async-dead-letters/{deadLetterId}/mark-handled")
    public ResponseEntity<BaseResult<DeadLetterResponse.Handled>> markHandled(
            @PathVariable String deadLetterId,
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            HttpServletRequest request) {
        requireCsrfToken(csrfToken);
        CurrentSession currentSession = requireAdminSession(request, sessionManager);
        DeadLetterHandledResult result = queryService.markHandled(currentSession.getAccountId(), deadLetterId);
        return ok(DeadLetterResponse.Handled.from(result));
    }

    private static void validatePage(int page, int pageSize) {
        if (page < 1) {
            throw new ValidationException("page", "最小值为 1");
        }
        if (pageSize < 1 || pageSize > 100) {
            throw new ValidationException("page_size", "取值范围为 1 到 100");
        }
    }

    private static DeadLetterSourceType parseSourceType(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Arrays.stream(DeadLetterSourceType.values())
                .filter(sourceType -> sourceType.getCode().equals(value))
                .findFirst()
                .orElseThrow(() -> new ValidationException("source_type", "不支持的死信来源类型"));
    }

    private static <T> ResponseEntity<BaseResult<T>> ok(T data) {
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(data));
    }
}
