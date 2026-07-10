package com.aiinterview.platform.interfaceadapter.http.task;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireAdminSession;
import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireCsrfToken;

import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.application.task.TaskAdminApplicationService;
import com.aiinterview.platform.application.task.command.AdminRetryTaskCommand;
import com.aiinterview.platform.application.task.dto.AdminFailedTaskPageResult;
import com.aiinterview.platform.application.task.dto.TaskFailureDetailResult;
import com.aiinterview.platform.application.task.dto.TaskStatusResult;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.domain.task.TaskType;
import com.aiinterview.platform.interfaceadapter.error.ValidationException;
import com.aiinterview.platform.interfaceadapter.http.task.request.RetryTaskRequest;
import com.aiinterview.platform.interfaceadapter.http.task.response.AdminFailedTaskPageResponse;
import com.aiinterview.platform.interfaceadapter.http.task.response.AdminFailureDetailResponse;
import com.aiinterview.platform.interfaceadapter.http.task.response.TaskStatusResponse;
import jakarta.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.UUID;
import org.springframework.http.CacheControl;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 管理员任务排查接口。
 */
@RestController
@RequestMapping("/api/v1/admin/tasks")
public class AdminTaskController {

    private final TaskAdminApplicationService taskAdminApplicationService;
    private final SessionManager sessionManager;

    public AdminTaskController(
            TaskAdminApplicationService taskAdminApplicationService,
            SessionManager sessionManager
    ) {
        this.taskAdminApplicationService = taskAdminApplicationService;
        this.sessionManager = sessionManager;
    }

    @GetMapping("/failed")
    public ResponseEntity<BaseResult<AdminFailedTaskPageResponse>> listFailedTasks(
            @RequestParam(name = "page", defaultValue = "1") int page,
            @RequestParam(name = "page_size", defaultValue = "20") int pageSize,
            @RequestParam(name = "task_type", required = false) String taskType,
            @RequestParam(name = "retryable", required = false) Boolean retryable,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireAdminSession(request, sessionManager);
        validatePage(page, pageSize);
        AdminFailedTaskPageResult result = taskAdminApplicationService.getFailedTaskList(
                currentSession.getAccountId(),
                currentSession.getRole().name(),
                parseTaskType(taskType),
                retryable,
                page,
                pageSize
        );
        return ok(AdminFailedTaskPageResponse.from(result));
    }

    @GetMapping("/{taskId}/failure")
    public ResponseEntity<BaseResult<AdminFailureDetailResponse>> getFailureDetail(
            @PathVariable String taskId,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireAdminSession(request, sessionManager);
        TaskFailureDetailResult result = taskAdminApplicationService.getFailureDetail(
                currentSession.getAccountId(),
                currentSession.getRole().name(),
                taskId
        );
        return ok(AdminFailureDetailResponse.from(result));
    }

    @PostMapping("/{taskId}/retry")
    public ResponseEntity<BaseResult<TaskStatusResponse>> retryTask(
            @PathVariable String taskId,
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "Idempotency-Key", required = false) String idempotencyKey,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            @RequestBody(required = false) RetryTaskRequest body,
            HttpServletRequest request
    ) {
        requireCsrfToken(csrfToken);
        requireIdempotencyKey(idempotencyKey);
        CurrentSession currentSession = requireAdminSession(request, sessionManager);
        String reason = body == null ? null : body.getReason();
        TaskStatusResult result = taskAdminApplicationService.adminRetryTask(new AdminRetryTaskCommand(
                currentSession.getAccountId(),
                currentSession.getRole().name(),
                taskId,
                reason,
                idempotencyKey,
                requestHash(taskId, reason),
                traceId(traceId)
        ));
        return ok(TaskStatusResponse.from(result));
    }

    private static void validatePage(int page, int pageSize) {
        if (page < 1) {
            throw new ValidationException("page", "最小值为 1");
        }
        if (pageSize < 1 || pageSize > 100) {
            throw new ValidationException("page_size", "取值范围为 1 到 100");
        }
    }

    private static TaskType parseTaskType(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Arrays.stream(TaskType.values())
                .filter(taskType -> taskType.getCode().equals(value))
                .findFirst()
                .orElseThrow(() -> new ValidationException("task_type", "不支持的任务类型"));
    }

    private static void requireIdempotencyKey(String idempotencyKey) {
        if (idempotencyKey == null || idempotencyKey.isBlank()) {
            throw new ValidationException("Idempotency-Key", "不能为空");
        }
    }

    private static String traceId(String traceId) {
        return traceId == null || traceId.isBlank() ? UUID.randomUUID().toString() : traceId;
    }

    private static String requestHash(String taskId, String reason) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest((taskId + ":" + (reason == null ? "" : reason)).getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("SHA-256 摘要算法不可用", ex);
        }
    }

    private static <T> ResponseEntity<BaseResult<T>> ok(T data) {
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(data));
    }
}
