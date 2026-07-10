package com.aiinterview.platform.interfaceadapter.http.file;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireAdminSession;

import com.aiinterview.platform.application.file.FileAdminQueryService;
import com.aiinterview.platform.application.file.dto.AdminFileStatusPageResult;
import com.aiinterview.platform.application.file.dto.AdminResultStatusPageResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.error.ValidationException;
import com.aiinterview.platform.interfaceadapter.http.file.response.AdminFileStatusPageResponse;
import com.aiinterview.platform.interfaceadapter.http.file.response.AdminResultStatusPageResponse;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.CacheControl;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 管理员文件与结果状态查询接口。
 */
@RestController
@RequestMapping("/api/v1/admin")
public class AdminFileController {

    private final FileAdminQueryService fileAdminQueryService;
    private final SessionManager sessionManager;

    public AdminFileController(FileAdminQueryService fileAdminQueryService, SessionManager sessionManager) {
        this.fileAdminQueryService = fileAdminQueryService;
        this.sessionManager = sessionManager;
    }

    @GetMapping("/files")
    public ResponseEntity<BaseResult<AdminFileStatusPageResponse>> listFiles(
            @RequestParam(name = "page", defaultValue = "1") int page,
            @RequestParam(name = "page_size", defaultValue = "20") int pageSize,
            @RequestParam(name = "file_status", required = false) String fileStatus,
            @RequestParam(name = "file_purpose", required = false) String filePurpose,
            @RequestParam(name = "task_id", required = false) String taskId,
            HttpServletRequest request
    ) {
        validatePage(page, pageSize);
        CurrentSession session = requireAdminSession(request, sessionManager);
        AdminFileStatusPageResult result = fileAdminQueryService.listFileStatuses(
                session.getAccountId(),
                fileStatus,
                filePurpose,
                taskId,
                page,
                pageSize
        );
        return ok(AdminFileStatusPageResponse.from(result));
    }

    @GetMapping("/task-results")
    public ResponseEntity<BaseResult<AdminResultStatusPageResponse>> listTaskResults(
            @RequestParam(name = "page", defaultValue = "1") int page,
            @RequestParam(name = "page_size", defaultValue = "20") int pageSize,
            @RequestParam(name = "task_id", required = false) String taskId,
            @RequestParam(name = "result_type", required = false) String resultType,
            @RequestParam(name = "attempt_no", required = false) Integer attemptNo,
            HttpServletRequest request
    ) {
        validatePage(page, pageSize);
        if (attemptNo != null && attemptNo < 1) {
            throw new ValidationException("attempt_no", "最小值为 1");
        }
        CurrentSession session = requireAdminSession(request, sessionManager);
        AdminResultStatusPageResult result = fileAdminQueryService.listResultStatuses(
                session.getAccountId(),
                taskId,
                resultType,
                attemptNo,
                page,
                pageSize
        );
        return ok(AdminResultStatusPageResponse.from(result));
    }

    private static void validatePage(int page, int pageSize) {
        if (page < 1) {
            throw new ValidationException("page", "最小值为 1");
        }
        if (pageSize < 1 || pageSize > 100) {
            throw new ValidationException("page_size", "取值范围为 1 到 100");
        }
    }

    private static <T> ResponseEntity<BaseResult<T>> ok(T data) {
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(data));
    }
}
