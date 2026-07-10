package com.aiinterview.platform.application.file;

import com.aiinterview.platform.application.file.command.BindFileToTaskCommand;
import com.aiinterview.platform.application.file.command.UploadFileCommand;
import com.aiinterview.platform.application.file.dto.FileBindingResult;
import com.aiinterview.platform.application.file.dto.FileUploadResult;
import com.aiinterview.platform.application.file.port.FileStoragePort;
import com.aiinterview.platform.application.security.FileContentInspectionResult;
import com.aiinterview.platform.application.security.FileContentInspector;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.ResourceOwnershipPort;
import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FileBindType;
import com.aiinterview.platform.domain.file.FileBinding;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.file.FileType;
import java.io.ByteArrayInputStream;
import java.time.Clock;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * 文件应用服务。
 */
public class FileApplicationService {

    private static final int FILE_PURPOSE_NOT_SUPPORTED = 72003;
    private static final int FILE_TYPE_PURPOSE_MISMATCH = 72004;
    private static final int FILE_SIZE_EXCEEDED = 72005;
    private static final int FILE_BIND_NOT_ALLOWED = 72006;
    private static final int FILE_STORAGE_FAILED = 72007;
    private static final int FILE_NOT_FOUND = 72001;

    private final FileRepository fileRepository;
    private final FileBindingRepository fileBindingRepository;
    private final FileStoragePort storagePort;
    private final FileContentInspector contentInspector;
    private final ResourceOwnershipPort resourceOwnershipPort;
    private final AuditWritePort auditWritePort;
    private final Supplier<String> idSupplier;
    private final Clock clock;

    public FileApplicationService(
            FileRepository fileRepository,
            FileBindingRepository fileBindingRepository,
            FileStoragePort storagePort,
            FileContentInspector contentInspector,
            ResourceOwnershipPort resourceOwnershipPort,
            AuditWritePort auditWritePort,
            Supplier<String> idSupplier,
            Clock clock
    ) {
        this.fileRepository = Objects.requireNonNull(fileRepository, "fileRepository 不能为空");
        this.fileBindingRepository = Objects.requireNonNull(fileBindingRepository, "fileBindingRepository 不能为空");
        this.storagePort = Objects.requireNonNull(storagePort, "storagePort 不能为空");
        this.contentInspector = Objects.requireNonNull(contentInspector, "contentInspector 不能为空");
        this.resourceOwnershipPort = Objects.requireNonNull(resourceOwnershipPort, "resourceOwnershipPort 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
    }

    public FileUploadResult uploadFile(UploadFileCommand command) {
        validateSourceDeviceId(command.sourceDeviceId());
        FilePurpose purpose = FilePurpose.fromCode(command.filePurpose())
                .orElseThrow(() -> new FileException(FILE_PURPOSE_NOT_SUPPORTED, "文件用途不支持"));
        FileContentInspectionResult inspection = contentInspector.inspect(
                command.originalName(),
                command.declaredMimeType(),
                command.content()
        );
        FileType fileType = inspection.fileType();
        if (!purpose.allows(fileType)) {
            throw new FileException(FILE_TYPE_PURPOSE_MISMATCH, "文件类型与用途不匹配");
        }
        if (command.content().length > purpose.getMaxSizeBytes()) {
            throw new FileException(FILE_SIZE_EXCEEDED, "文件大小超限");
        }

        Instant now = Instant.now(clock);
        String fileId = idSupplier.get();
        String storedFilename = fileId + "." + fileType.getCode();
        String objectKey = command.userId() + "/" + storedFilename;
        File file = File.createUploading(
                fileId,
                command.userId(),
                command.originalName(),
                storedFilename,
                fileType,
                purpose,
                command.content().length,
                command.declaredMimeType(),
                command.sourceDeviceId(),
                objectKey,
                now
        );
        fileRepository.save(file);

        try {
            storagePort.store(objectKey, new ByteArrayInputStream(command.content()), command.content().length);
        } catch (RuntimeException ex) {
            file.markFailed("文件存储失败", Instant.now(clock));
            fileRepository.updateStatus(file);
            auditWritePort.writeAudit("FILE_UPLOAD_FAILED", command.userId(), fileId, Map.of(
                    "reason", "文件存储失败",
                    "trace_id", command.traceId()
            ));
            throw new FileException(FILE_STORAGE_FAILED, "文件存储失败", ex);
        }
        file.markAvailable(null, now);
        fileRepository.updateStatus(file);
        auditWritePort.writeAudit("FILE_UPLOADED", command.userId(), fileId, Map.of(
                "file_type", fileType.getCode(),
                "file_size", command.content().length,
                "trace_id", command.traceId()
        ));
        return new FileUploadResult(
                file.getId(),
                file.getOriginalName(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode(),
                file.getFileSize(),
                file.getFileStatus().getCode(),
                file.getCreatedAt()
        );
    }

    public FileBindingResult bindFileToTask(BindFileToTaskCommand command) {
        if (command.attemptNo() < 0) {
            throw new FileException(FILE_BIND_NOT_ALLOWED, "文件不可用于当前任务绑定");
        }
        FileBindType bindType = parseBindType(command.bindType());
        if (bindType != FileBindType.INPUT) {
            throw new FileException(FILE_BIND_NOT_ALLOWED, "用户入口不得创建产物绑定");
        }
        File file = fileRepository.findById(command.fileId())
                .orElseThrow(() -> new FileException(FILE_NOT_FOUND, "文件不存在"));
        try {
            file.assertOwnedBy(command.currentUserId());
        } catch (RuntimeException ex) {
            throw new FileException(FILE_NOT_FOUND, "文件不存在");
        }
        if (!file.canBindAsInput()) {
            throw new FileException(FILE_BIND_NOT_ALLOWED, "文件不可用于当前任务绑定");
        }
        if (!resourceOwnershipPort.isOwnedByUser(command.currentUserId(), "task", command.taskId())) {
            throw new FileException(FILE_BIND_NOT_ALLOWED, "文件不可用于当前任务绑定");
        }

        FileBinding.FileBindingUniqueKey uniqueKey = new FileBinding.FileBindingUniqueKey(
                command.fileId(),
                command.taskId(),
                bindType,
                command.attemptNo()
        );
        return fileBindingRepository.findByUniqueKey(uniqueKey)
                .map(this::toBindingResult)
                .orElseGet(() -> createInputBinding(command));
    }

    private FileBindingResult createInputBinding(BindFileToTaskCommand command) {
        FileBinding binding = FileBinding.createInput(
                idSupplier.get(),
                command.fileId(),
                command.taskId(),
                command.currentUserId(),
                command.attemptNo(),
                Instant.now(clock)
        );
        fileBindingRepository.save(binding);
        auditWritePort.writeAudit("FILE_BOUND_TO_TASK", command.currentUserId(), command.fileId(), Map.of(
                "task_id", command.taskId(),
                "bind_type", binding.getBindType().getCode(),
                "trace_id", command.traceId()
        ));
        return toBindingResult(binding);
    }

    private FileBindingResult toBindingResult(FileBinding binding) {
        return new FileBindingResult(
                binding.getId(),
                binding.getFileId(),
                binding.getTaskId(),
                binding.getBindType().getCode(),
                binding.getAttemptNo()
        );
    }

    private FileBindType parseBindType(String bindType) {
        if ("input".equals(bindType)) {
            return FileBindType.INPUT;
        }
        if ("output".equals(bindType)) {
            return FileBindType.OUTPUT;
        }
        if ("report".equals(bindType)) {
            return FileBindType.REPORT;
        }
        throw new FileException(FILE_BIND_NOT_ALLOWED, "文件不可用于当前任务绑定");
    }

    private static void validateSourceDeviceId(String sourceDeviceId) {
        if (sourceDeviceId == null || sourceDeviceId.isBlank()) {
            return;
        }
        if (sourceDeviceId.length() > 128 || !sourceDeviceId.matches("[A-Za-z0-9._:-]+")) {
            throw new FileException(10001, "请求参数校验失败");
        }
    }
}
