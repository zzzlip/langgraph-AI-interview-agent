package com.aiinterview.platform.interfaceadapter.http.file;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.file.FileQueryService;
import com.aiinterview.platform.application.file.dto.FileTaskFilePageResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.error.ValidationException;
import com.aiinterview.platform.interfaceadapter.http.file.response.TaskFilePageResponse;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.CacheControl;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 任务关联文件查询接口。
 */
@RestController
@RequestMapping("/api/v1/tasks")
public class TaskFileController {

    private final FileQueryService fileQueryService;
    private final SessionManager sessionManager;

    public TaskFileController(FileQueryService fileQueryService, SessionManager sessionManager) {
        this.fileQueryService = fileQueryService;
        this.sessionManager = sessionManager;
    }

    @GetMapping("/{taskId}/files")
    public ResponseEntity<BaseResult<TaskFilePageResponse>> listTaskFiles(
            @PathVariable String taskId,
            @RequestParam(name = "bind_type", required = false) String bindType,
            @RequestParam(name = "page", defaultValue = "1") int page,
            @RequestParam(name = "page_size", defaultValue = "20") int pageSize,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        validatePage(page, pageSize);
        FileTaskFilePageResult result = fileQueryService.listTaskFiles(
                currentSession.getAccountId(),
                taskId,
                bindType,
                page,
                pageSize
        );
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(TaskFilePageResponse.from(result)));
    }

    private static void validatePage(int page, int pageSize) {
        if (page < 1) {
            throw new ValidationException("page", "最小值为 1");
        }
        if (pageSize < 1 || pageSize > 100) {
            throw new ValidationException("page_size", "取值范围为 1 到 100");
        }
    }
}
