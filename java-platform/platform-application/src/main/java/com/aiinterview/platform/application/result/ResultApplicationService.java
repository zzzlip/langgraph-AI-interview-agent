package com.aiinterview.platform.application.result;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.BindResultReferenceCommand;
import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FileBindType;
import com.aiinterview.platform.domain.file.FileBinding;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.file.FileType;
import com.aiinterview.platform.domain.result.ResultStatus;
import com.aiinterview.platform.domain.result.ResultType;
import com.aiinterview.platform.domain.result.TaskResult;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * 结果应用服务。
 */
public class ResultApplicationService {

    private static final int RESULT_ALREADY_EXISTS = 73002;
    private static final int RESULT_SAVE_FAILED = 73003;
    private static final int OUTPUT_METADATA_INCOMPLETE = 73004;
    private static final int RESULT_STATUS_NOT_SUPPORTED = 73005;

    private static final TypeReference<Map<String, Object>> SUMMARY_TYPE = new TypeReference<>() {
    };
    private static final Comparator<OutputFileSignature> OUTPUT_FILE_SIGNATURE_ORDER = Comparator
            .comparing(OutputFileSignature::objectKey)
            .thenComparing(OutputFileSignature::filePurpose)
            .thenComparing(OutputFileSignature::fileType)
            .thenComparing(OutputFileSignature::mimeType)
            .thenComparingLong(OutputFileSignature::fileSize)
            .thenComparing(OutputFileSignature::displayName);

    private final TaskResultRepository taskResultRepository;
    private final FileRepository fileRepository;
    private final FileBindingRepository fileBindingRepository;
    private final AuditWritePort auditWritePort;
    private final Supplier<String> idSupplier;
    private final Clock clock;
    private final ObjectMapper objectMapper;

    public ResultApplicationService(
            TaskResultRepository taskResultRepository,
            FileRepository fileRepository,
            FileBindingRepository fileBindingRepository,
            AuditWritePort auditWritePort,
            Supplier<String> idSupplier,
            Clock clock,
            ObjectMapper objectMapper
    ) {
        this.taskResultRepository = Objects.requireNonNull(taskResultRepository, "taskResultRepository 不能为空");
        this.fileRepository = Objects.requireNonNull(fileRepository, "fileRepository 不能为空");
        this.fileBindingRepository = Objects.requireNonNull(fileBindingRepository, "fileBindingRepository 不能为空");
        this.auditWritePort = Objects.requireNonNull(auditWritePort, "auditWritePort 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    public String saveCompletedResult(BindResultReferenceCommand command) {
        ResultStatus status = ResultStatus.fromCode(command.resultStatus())
                .orElseThrow(() -> fail(command, RESULT_STATUS_NOT_SUPPORTED, "结果状态不支持"));
        if (status != ResultStatus.COMPLETED) {
            throw fail(command, RESULT_STATUS_NOT_SUPPORTED, "结果状态不支持");
        }
        ResultType resultType = ResultType.fromCode(command.resultType())
                .orElseThrow(() -> fail(command, OUTPUT_METADATA_INCOMPLETE, "结果类型不支持"));
        Map<String, Object> summary = parseSummary(command);

        return taskResultRepository.findByTaskIdAndAttemptNo(command.taskId(), command.attemptNo())
                .map(existing -> handleExisting(command, existing, resultType, summary))
                .orElseGet(() -> createResult(command, resultType, summary));
    }

    private String createResult(BindResultReferenceCommand command, ResultType resultType, Map<String, Object> summary) {
        try {
            for (BindResultReferenceCommand.OutputFilePayload outputFile : command.outputFiles()) {
                saveOutputFileAndBinding(command, outputFile);
            }
            TaskResult result = TaskResult.createCompleted(
                    idSupplier.get(),
                    command.taskId(),
                    command.attemptNo(),
                    resultType,
                    summary,
                    Instant.now(clock)
            );
            taskResultRepository.save(result);
            auditWritePort.writeAudit("RESULT_SAVED", command.userId(), command.taskId(), Map.of(
                    "result_id", result.getId(),
                    "trace_id", command.traceId()
            ));
            return result.getId();
        } catch (ResultException ex) {
            throw ex;
        } catch (RuntimeException ex) {
            auditWritePort.writeAudit("RESULT_SAVE_FAILED", command.userId(), command.taskId(), Map.of(
                    "reason", "结果保存失败",
                    "trace_id", command.traceId()
            ));
            throw new ResultException(RESULT_SAVE_FAILED, "结果保存失败", ex);
        }
    }

    private void saveOutputFileAndBinding(
            BindResultReferenceCommand command,
            BindResultReferenceCommand.OutputFilePayload outputFile
    ) {
        ValidatedOutputFile validated = validateOutputFile(command, outputFile);
        File existingFile = findExistingOutputFile(command, validated);
        if (existingFile != null) {
            return;
        }

        Instant now = Instant.now(clock);
        String fileId = idSupplier.get();
        File file = File.createUploading(
                fileId,
                command.userId(),
                outputFile.displayName(),
                storedFilename(outputFile.objectKey()),
                validated.fileType(),
                validated.purpose(),
                outputFile.fileSize(),
                outputFile.mimeType(),
                outputFile.objectKey(),
                now
        );
        file.markAvailable(null, now);
        fileRepository.save(file);
        FileBinding binding = FileBinding.createWorkerOutput(
                idSupplier.get(),
                file.getId(),
                command.taskId(),
                command.userId(),
                validated.bindType(),
                command.attemptNo(),
                now
        );
        fileBindingRepository.save(binding);
    }

    private File findExistingOutputFile(BindResultReferenceCommand command, ValidatedOutputFile validated) {
        return existingOutputFiles(command.taskId(), command.attemptNo())
                .stream()
                .filter(file -> file.getObjectKey().equals(validated.signature().objectKey()))
                .map(file -> {
                    if (!toSignature(file).equals(validated.signature())) {
                        throw fail(command, RESULT_ALREADY_EXISTS, "结果已存在");
                    }
                    return file;
                })
                .findFirst()
                .orElse(null);
    }

    private String handleExisting(
            BindResultReferenceCommand command,
            TaskResult existing,
            ResultType resultType,
            Map<String, Object> summary
    ) {
        if (existing.getResultType() == resultType
                && existing.getSummary().equals(summary)
                && existingOutputSignatures(existing).equals(commandOutputSignatures(command))) {
            return existing.getId();
        }
        throw fail(command, RESULT_ALREADY_EXISTS, "结果已存在");
    }

    private List<OutputFileSignature> commandOutputSignatures(BindResultReferenceCommand command) {
        return command.outputFiles().stream()
                .map(outputFile -> validateOutputFile(command, outputFile).signature())
                .sorted(OUTPUT_FILE_SIGNATURE_ORDER)
                .toList();
    }

    private List<OutputFileSignature> existingOutputSignatures(TaskResult existing) {
        return existingOutputFiles(existing.getTaskId(), existing.getAttemptNo()).stream()
                .map(this::toSignature)
                .sorted(OUTPUT_FILE_SIGNATURE_ORDER)
                .toList();
    }

    private List<File> existingOutputFiles(String taskId, int attemptNo) {
        List<File> files = new ArrayList<>();
        files.addAll(existingOutputFiles(taskId, attemptNo, FileBindType.REPORT));
        files.addAll(existingOutputFiles(taskId, attemptNo, FileBindType.OUTPUT));
        return files;
    }

    private List<File> existingOutputFiles(String taskId, int attemptNo, FileBindType bindType) {
        return fileBindingRepository.findByTaskIdAndBindType(taskId, bindType)
                .stream()
                .filter(binding -> binding.getAttemptNo() == attemptNo)
                .map(binding -> fileRepository.findById(binding.getFileId()).orElse(null))
                .filter(Objects::nonNull)
                .toList();
    }

    private ValidatedOutputFile validateOutputFile(
            BindResultReferenceCommand command,
            BindResultReferenceCommand.OutputFilePayload outputFile
    ) {
        FileType fileType = FileType.fromCode(outputFile.fileType())
                .orElseThrow(() -> fail(command, OUTPUT_METADATA_INCOMPLETE, "Worker 产物元信息不完整"));
        FilePurpose purpose = FilePurpose.fromCode(outputFile.filePurpose())
                .orElseThrow(() -> fail(command, OUTPUT_METADATA_INCOMPLETE, "Worker 产物元信息不完整"));
        if (purpose.isUserUpload() || !purpose.allows(fileType) || !fileType.getMimeType().equals(outputFile.mimeType())) {
            throw fail(command, OUTPUT_METADATA_INCOMPLETE, "Worker 产物元信息不完整");
        }
        OutputFileSignature signature = new OutputFileSignature(
                outputFile.objectKey(),
                outputFile.fileType(),
                outputFile.filePurpose(),
                outputFile.mimeType(),
                outputFile.fileSize(),
                outputFile.displayName()
        );
        return new ValidatedOutputFile(fileType, purpose, outputBindType(purpose), signature);
    }

    private OutputFileSignature toSignature(File file) {
        return new OutputFileSignature(
                file.getObjectKey(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode(),
                file.getMimeType(),
                file.getFileSize(),
                file.getOriginalName()
        );
    }

    private Map<String, Object> parseSummary(BindResultReferenceCommand command) {
        try {
            Map<String, Object> summary = objectMapper.readValue(command.summary(), SUMMARY_TYPE);
            if (!(summary.get("schema_version") instanceof Integer)) {
                throw fail(command, OUTPUT_METADATA_INCOMPLETE, "Worker 产物元信息不完整");
            }
            return Map.copyOf(new LinkedHashMap<>(summary));
        } catch (ResultException ex) {
            throw ex;
        } catch (Exception ex) {
            throw fail(command, OUTPUT_METADATA_INCOMPLETE, "Worker 产物元信息不完整");
        }
    }

    private ResultException fail(BindResultReferenceCommand command, int code, String message) {
        auditWritePort.writeAudit("RESULT_SAVE_FAILED", command.userId(), command.taskId(), Map.of(
                "reason", message,
                "trace_id", command.traceId()
        ));
        return new ResultException(code, message);
    }

    private static FileBindType outputBindType(FilePurpose purpose) {
        if (purpose == FilePurpose.EVALUATION_REPORT || purpose == FilePurpose.INTERVIEW_REPORT) {
            return FileBindType.REPORT;
        }
        return FileBindType.OUTPUT;
    }

    private static String storedFilename(String objectKey) {
        int slashIndex = objectKey.lastIndexOf('/');
        if (slashIndex >= 0 && slashIndex < objectKey.length() - 1) {
            return objectKey.substring(slashIndex + 1);
        }
        return objectKey;
    }

    private record ValidatedOutputFile(
            FileType fileType,
            FilePurpose purpose,
            FileBindType bindType,
            OutputFileSignature signature
    ) {
    }

    private record OutputFileSignature(
            String objectKey,
            String fileType,
            String filePurpose,
            String mimeType,
            long fileSize,
            String displayName
    ) {
    }
}
