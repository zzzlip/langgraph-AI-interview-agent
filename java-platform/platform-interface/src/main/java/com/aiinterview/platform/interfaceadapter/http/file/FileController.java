package com.aiinterview.platform.interfaceadapter.http.file;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireCsrfToken;
import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.file.FileApplicationService;
import com.aiinterview.platform.application.file.FileQueryService;
import com.aiinterview.platform.application.file.command.UploadFileCommand;
import com.aiinterview.platform.application.file.dto.FileContentResult;
import com.aiinterview.platform.application.file.dto.FileDetailResult;
import com.aiinterview.platform.application.file.dto.FileUploadResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.error.ValidationException;
import com.aiinterview.platform.interfaceadapter.http.file.response.FileResponse;
import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.io.InputStream;
import java.util.UUID;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.CacheControl;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * 普通用户文件接口。
 */
@RestController
@RequestMapping("/api/v1/files")
public class FileController {

    private final FileApplicationService fileApplicationService;
    private final FileQueryService fileQueryService;
    private final SessionManager sessionManager;

    public FileController(
            FileApplicationService fileApplicationService,
            FileQueryService fileQueryService,
            SessionManager sessionManager
    ) {
        this.fileApplicationService = fileApplicationService;
        this.fileQueryService = fileQueryService;
        this.sessionManager = sessionManager;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<BaseResult<FileResponse>> uploadFile(
            @RequestParam("file") MultipartFile file,
            @RequestParam("file_purpose") String filePurpose,
            @RequestParam(name = "source_device_id", required = false) String sourceDeviceId,
            @RequestHeader(name = "X-CSRF-Token", required = false) String csrfToken,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            HttpServletRequest request
    ) throws IOException {
        requireCsrfToken(csrfToken);
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        if (file.isEmpty()) {
            throw new ValidationException("file", "不能为空");
        }
        FileUploadResult result = fileApplicationService.uploadFile(new UploadFileCommand(
                currentSession.getAccountId(),
                file.getOriginalFilename(),
                filePurpose,
                contentType(file),
                file.getBytes(),
                sourceDeviceId,
                traceId(traceId)
        ));
        return ResponseEntity.status(201)
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(FileResponse.from(result)));
    }

    @GetMapping("/{fileId}")
    public ResponseEntity<BaseResult<FileResponse>> getFileDetail(
            @PathVariable String fileId,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        FileDetailResult result = fileQueryService.getFileDetail(currentSession.getAccountId(), fileId);
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(FileResponse.from(result)));
    }

    @GetMapping("/{fileId}/download")
    public ResponseEntity<InputStreamResource> download(
            @PathVariable String fileId,
            @RequestHeader(name = "X-Trace-Id", required = false) String traceId,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        FileContentResult result = fileQueryService.download(currentSession.getAccountId(), fileId, traceId(traceId));
        return contentResponse(result, ContentDisposition.attachment().filename(result.fileName()).build());
    }

    @GetMapping("/{fileId}/preview")
    public ResponseEntity<InputStreamResource> preview(
            @PathVariable String fileId,
            HttpServletRequest request
    ) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        FileContentResult result = fileQueryService.preview(currentSession.getAccountId(), fileId);
        return contentResponse(result, ContentDisposition.inline().filename(result.fileName()).build());
    }

    private static ResponseEntity<InputStreamResource> contentResponse(FileContentResult result, ContentDisposition disposition) {
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType(result.mimeType()))
                .contentLength(result.fileSize())
                .header(HttpHeaders.CACHE_CONTROL, result.cacheControl())
                .header(HttpHeaders.PRAGMA, "no-cache")
                .header(HttpHeaders.CONTENT_DISPOSITION, disposition.toString())
                .body(new InputStreamResource(nonNullContent(result.content())));
    }

    private static InputStream nonNullContent(InputStream content) {
        if (content == null) {
            throw new IllegalStateException("文件内容不能为空");
        }
        return content;
    }

    private static String contentType(MultipartFile file) {
        String contentType = file.getContentType();
        return contentType == null || contentType.isBlank() ? MediaType.APPLICATION_OCTET_STREAM_VALUE : contentType;
    }

    private static String traceId(String traceId) {
        return traceId == null || traceId.isBlank() ? UUID.randomUUID().toString() : traceId;
    }
}
