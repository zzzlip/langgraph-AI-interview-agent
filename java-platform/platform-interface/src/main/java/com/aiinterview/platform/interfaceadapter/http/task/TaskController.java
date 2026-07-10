package com.aiinterview.platform.interfaceadapter.http.task;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireCsrfToken;
import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.application.task.TaskApplicationService;
import com.aiinterview.platform.application.task.TaskQueryService;
import com.aiinterview.platform.application.task.command.CancelTaskCommand;
import com.aiinterview.platform.application.task.command.RetryTaskCommand;
import com.aiinterview.platform.application.task.dto.TaskDetailResult;
import com.aiinterview.platform.application.task.dto.TaskPageResult;
import com.aiinterview.platform.application.task.dto.TaskStatusResult;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.aiinterview.platform.domain.task.TaskType;
import com.aiinterview.platform.interfaceadapter.error.ValidationException;
import com.aiinterview.platform.interfaceadapter.http.task.request.RetryTaskRequest;
import com.aiinterview.platform.interfaceadapter.http.task.response.TaskDetailResponse;
import com.aiinterview.platform.interfaceadapter.http.task.response.TaskPageResponse;
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
 * 普通用户任务接口。
 */
@RestController
@RequestMapping("/api/v1/tasks")
public class TaskController {

    private final TaskQueryService taskQueryService;
    private final TaskApplicationService taskApplicationService;
    private final SessionManager sessionManager;

    public TaskController(
            TaskQueryService taskQueryService,
            TaskApplicationService taskApplicationService,
            SessionManager sessionManager
    ) {
        this.taskQueryService = taskQueryService;
        this.taskApplicationService = taskApplicationService;
        this.sessionManager = sessionManager;
    }

    @GetMapping
    public ResponseEntity<BaseResult<TaskPageResponse>> listTasks(
            @RequestParam(name = "page", defaultValue = "1") int page,
            @RequestParam(name = "page_size", defaultValue = "20") int pageSize,
            @RequestParam(name = "task_type", required = false) String taskType,
            @RequestParam(name = "status", required = false) String status,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        validatePage(page, pageSize);
        TaskPageResult result = taskQueryService.getTaskList(
                currentSession.getAccountId(),
                parseTaskType(taskType),
                parseTaskStatus(status),
                page,
                pageSize
        );
        return ok(TaskPageResponse.from(result));
    }

    @GetMapping("/{taskId}")
    public ResponseEntity<BaseResult<TaskDetailResponse>> getTaskDetail(
            @PathVariable String taskId,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        TaskDetailResult result = taskQueryService.getTaskDetail(currentSession.getAccountId(), taskId);
        return ok(TaskDetailResponse.from(result));
    }

    @PostMapping("/{taskId}/cancel")
    public ResponseEntity<BaseResult<TaskStatusResponse>> cancelTask(
            @PathVariable String taskId,
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            HttpServletRequest request
    ) {
        requireCsrfToken(csrfToken);
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        TaskStatusResult result = taskApplicationService.cancelTask(new CancelTaskCommand(
                currentSession.getAccountId(),
                taskId,
                traceId(traceId)
        ));
        return ok(TaskStatusResponse.from(result));
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
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        String reason = body == null ? null : body.getReason();
        TaskStatusResult result = taskApplicationService.retryTask(new RetryTaskCommand(
                currentSession.getAccountId(),
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

    private static void requireIdempotencyKey(String idempotencyKey) {
        if (idempotencyKey == null || idempotencyKey.isBlank()) {
            throw new ValidationException("Idempotency-Key", "不能为空");
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

    private static TaskStatus parseTaskStatus(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Arrays.stream(TaskStatus.values())
                .filter(taskStatus -> taskStatus.getCode().equals(value))
                .findFirst()
                .orElseThrow(() -> new ValidationException("status", "不支持的任务状态"));
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
