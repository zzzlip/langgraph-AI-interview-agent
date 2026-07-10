package com.aiinterview.platform.application.file;

import com.aiinterview.platform.application.file.dto.FileContentResult;
import com.aiinterview.platform.application.file.dto.FileDetailResult;
import com.aiinterview.platform.application.file.dto.FileTaskFileItem;
import com.aiinterview.platform.application.file.dto.FileTaskFilePageResult;
import com.aiinterview.platform.application.file.port.FileStoragePort;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.ResourceOwnershipPort;
import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FileBindType;
import com.aiinterview.platform.domain.file.FileBinding;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.file.FileStatus;
import com.aiinterview.platform.domain.file.FileType;
import com.aiinterview.platform.domain.task.PageResult;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 文件查询应用服务。
 */
public class FileQueryService {

    private static final int FILE_NOT_FOUND = 72001;
    private static final int FILE_READ_FAILED = 72008;
    private static final int FILE_PREVIEW_NOT_SUPPORTED = 72101;
    private static final String NO_STORE_CACHE_CONTROL = "no-store, must-revalidate";

    private final FileRepository fileRepository;
    private final FileBindingRepository fileBindingRepository;
    private final FileStoragePort storagePort;
    private final ResourceOwnershipPort resourceOwnershipPort;
    private final AuditWritePort auditWritePort;

    public FileQueryService(FileRepository fileRepository, FileStoragePort storagePort, AuditWritePort auditWritePort) {
        this(fileRepository, null, storagePort, null, auditWritePort);
    }

    public FileQueryService(
            FileRepository fileRepository,
            FileBindingRepository fileBindingRepository,
            FileStoragePort storagePort,
            ResourceOwnershipPort resourceOwnershipPort,
            AuditWritePort auditWritePort
    ) {
        this.fileRepository = Objects.requireNonNull(fileRepository, "fileRepository 不能为空");
        this.fileBindingRepository = fileBindingRepository;
        this.storagePort = Objects.requireNonNull(storagePort, "storagePort 不能为空");
        this.resourceOwnershipPort = resourceOwnershipPort;
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
    }

    public FileDetailResult getFileDetail(String currentUserId, String fileId) {
        File file = loadOwnedFile(currentUserId, fileId);
        return new FileDetailResult(
                file.getId(),
                file.getOriginalName(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode(),
                file.getFileSize(),
                file.getFileStatus().getCode(),
                file.getCreatedAt()
        );
    }

    public FileContentResult download(String currentUserId, String fileId, String traceId) {
        File file = loadAvailableOwnedFile(currentUserId, fileId);
        InputStream content = retrieve(file);
        auditWritePort.writeAudit("FILE_DOWNLOADED", currentUserId, fileId, Map.of("trace_id", traceId));
        return toContentResult(file, false, content);
    }

    public FileContentResult preview(String currentUserId, String fileId) {
        File file = loadAvailableOwnedFile(currentUserId, fileId);
        if (!isPreviewSupported(file.getFileType())) {
            throw new FileException(FILE_PREVIEW_NOT_SUPPORTED, "该文件类型不支持预览");
        }
        return toContentResult(file, true, retrieve(file));
    }

    public FileTaskFilePageResult listTaskFiles(String currentUserId, String taskId, String bindType, int page, int pageSize) {
        ensureTaskFileDependencies();
        if (!resourceOwnershipPort.isOwnedByUser(currentUserId, "task", taskId)) {
            throw new FileException(71003, "任务不存在");
        }
        PageResult<FileBinding> bindings = loadTaskBindings(taskId, bindType, page, pageSize);
        Map<String, File> filesById = loadFilesById(currentUserId, bindings.items());
        List<FileTaskFileItem> items = bindings.items().stream()
                .map(binding -> toTaskFileItem(binding, filesById))
                .toList();
        return new FileTaskFilePageResult(items, bindings.total(), bindings.page(), bindings.pageSize());
    }

    private File loadOwnedFile(String currentUserId, String fileId) {
        File file = fileRepository.findById(fileId)
                .orElseThrow(() -> new FileException(FILE_NOT_FOUND, "文件不存在"));
        try {
            file.assertOwnedBy(currentUserId);
        } catch (RuntimeException ex) {
            throw new FileException(FILE_NOT_FOUND, "文件不存在");
        }
        return file;
    }

    private File loadAvailableOwnedFile(String currentUserId, String fileId) {
        File file = loadOwnedFile(currentUserId, fileId);
        if (file.getFileStatus() != FileStatus.AVAILABLE) {
            throw new FileException(FILE_NOT_FOUND, "文件不存在");
        }
        return file;
    }

    private InputStream retrieve(File file) {
        try {
            return storagePort.retrieve(file.getObjectKey());
        } catch (RuntimeException ex) {
            throw new FileException(FILE_READ_FAILED, "文件读取失败", ex);
        }
    }

    private static FileContentResult toContentResult(File file, boolean inline, InputStream content) {
        return new FileContentResult(
                file.getId(),
                file.getOriginalName(),
                file.getMimeType(),
                file.getFileSize(),
                NO_STORE_CACHE_CONTROL,
                inline,
                content
        );
    }

    private PageResult<FileBinding> loadTaskBindings(String taskId, String bindType, int page, int pageSize) {
        if (bindType == null || bindType.isBlank()) {
            return fileBindingRepository.findByTaskId(taskId, page, pageSize);
        }
        return fileBindingRepository.findByTaskIdAndBindType(taskId, parseBindType(bindType), page, pageSize);
    }

    private Map<String, File> loadFilesById(String currentUserId, List<FileBinding> bindings) {
        List<String> fileIds = bindings.stream()
                .map(FileBinding::getFileId)
                .distinct()
                .toList();
        Map<String, File> filesById = new HashMap<>();
        for (File file : fileRepository.findByIds(fileIds)) {
            try {
                file.assertOwnedBy(currentUserId);
            } catch (RuntimeException ex) {
                throw new FileException(FILE_NOT_FOUND, "文件不存在");
            }
            filesById.put(file.getId(), file);
        }
        return filesById;
    }

    private FileTaskFileItem toTaskFileItem(FileBinding binding, Map<String, File> filesById) {
        File file = filesById.get(binding.getFileId());
        if (file == null) {
            throw new FileException(FILE_NOT_FOUND, "文件不存在");
        }
        return new FileTaskFileItem(
                file.getId(),
                file.getOriginalName(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode(),
                file.getFileSize(),
                binding.getBindType().getCode(),
                binding.getCreatedAt()
        );
    }

    private static FileBindType parseBindType(String bindType) {
        for (FileBindType value : FileBindType.values()) {
            if (value.getCode().equals(bindType)) {
                return value;
            }
        }
        throw new FileException(10001, "请求参数校验失败");
    }

    private void ensureTaskFileDependencies() {
        if (fileBindingRepository == null || resourceOwnershipPort == null) {
            throw new IllegalStateException("任务关联文件查询依赖未配置");
        }
    }

    private static boolean isPreviewSupported(FileType fileType) {
        return fileType == FileType.PNG || fileType == FileType.JPG || fileType == FileType.JPEG || fileType == FileType.PDF;
    }
}
