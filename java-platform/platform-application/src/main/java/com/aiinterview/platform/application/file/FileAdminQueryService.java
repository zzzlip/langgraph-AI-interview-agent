package com.aiinterview.platform.application.file;

import com.aiinterview.platform.application.file.dto.AdminFileStatusPageResult;
import com.aiinterview.platform.application.file.dto.AdminResultStatusPageResult;
import com.aiinterview.platform.application.file.port.AdminFileStatusQuery;
import com.aiinterview.platform.application.file.port.AdminFileStatusQueryPort;
import com.aiinterview.platform.application.file.port.AdminResultStatusQuery;
import com.aiinterview.platform.application.file.port.AdminResultStatusQueryPort;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileStatus;
import com.aiinterview.platform.domain.result.ResultType;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 管理员文件与结果状态查询服务。
 */
public class FileAdminQueryService {

    private static final int PARAMETER_INVALID = 10001;

    private final AdminFileStatusQueryPort fileStatusQueryPort;
    private final AdminResultStatusQueryPort resultStatusQueryPort;
    private final AuditWritePort auditWritePort;

    public FileAdminQueryService(
            AdminFileStatusQueryPort fileStatusQueryPort,
            AdminResultStatusQueryPort resultStatusQueryPort,
            AuditWritePort auditWritePort
    ) {
        this.fileStatusQueryPort = Objects.requireNonNull(fileStatusQueryPort, "fileStatusQueryPort 不能为空");
        this.resultStatusQueryPort = Objects.requireNonNull(resultStatusQueryPort, "resultStatusQueryPort 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
    }

    public AdminFileStatusPageResult listFileStatuses(
            String adminUserId,
            String fileStatus,
            String filePurpose,
            String taskId,
            int page,
            int pageSize
    ) {
        String parsedStatus = parseFileStatus(fileStatus);
        String parsedPurpose = parseFilePurpose(filePurpose);
        AdminFileStatusQuery query = new AdminFileStatusQuery(parsedStatus, parsedPurpose, blankToNull(taskId), page, pageSize);
        AdminFileStatusPageResult result = fileStatusQueryPort.search(query);
        auditWritePort.writeAudit("ADMIN_FILE_STATUS_VIEWED", adminUserId, "file", filters(query));
        return result;
    }

    public AdminResultStatusPageResult listResultStatuses(
            String adminUserId,
            String taskId,
            String resultType,
            Integer attemptNo,
            int page,
            int pageSize
    ) {
        String parsedResultType = parseResultType(resultType);
        AdminResultStatusQuery query = new AdminResultStatusQuery(blankToNull(taskId), parsedResultType, attemptNo, page, pageSize);
        AdminResultStatusPageResult result = resultStatusQueryPort.search(query);
        auditWritePort.writeAudit("ADMIN_RESULT_STATUS_VIEWED", adminUserId, "task_result", filters(query));
        return result;
    }

    private static String parseFileStatus(String value) {
        String text = blankToNull(value);
        if (text == null) {
            return null;
        }
        FileStatus.fromCode(text)
                .orElseThrow(() -> new FileException(PARAMETER_INVALID, "请求参数校验失败"));
        return text;
    }

    private static String parseFilePurpose(String value) {
        String text = blankToNull(value);
        if (text == null) {
            return null;
        }
        return FilePurpose.fromCode(text)
                .map(FilePurpose::getCode)
                .orElseThrow(() -> new FileException(PARAMETER_INVALID, "请求参数校验失败"));
    }

    private static String parseResultType(String value) {
        String text = blankToNull(value);
        if (text == null) {
            return null;
        }
        return ResultType.fromCode(text)
                .map(ResultType::getCode)
                .orElseThrow(() -> new FileException(PARAMETER_INVALID, "请求参数校验失败"));
    }

    private static Map<String, Object> filters(AdminFileStatusQuery query) {
        Map<String, Object> details = new LinkedHashMap<>();
        putIfPresent(details, "file_status", query.fileStatus());
        putIfPresent(details, "file_purpose", query.filePurpose());
        putIfPresent(details, "task_id", query.taskId());
        details.put("page", query.page());
        details.put("page_size", query.pageSize());
        return details;
    }

    private static Map<String, Object> filters(AdminResultStatusQuery query) {
        Map<String, Object> details = new LinkedHashMap<>();
        putIfPresent(details, "task_id", query.taskId());
        putIfPresent(details, "result_type", query.resultType());
        putIfPresent(details, "attempt_no", query.attemptNo());
        details.put("page", query.page());
        details.put("page_size", query.pageSize());
        return details;
    }

    private static void putIfPresent(Map<String, Object> details, String key, Object value) {
        if (value != null) {
            details.put(key, value);
        }
    }

    private static String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value;
    }
}
