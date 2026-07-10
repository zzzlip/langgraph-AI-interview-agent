package com.aiinterview.platform.application.resumeoptimization;

import com.aiinterview.platform.application.resumeoptimization.command.CreateResumeOptimizationCommand;
import com.aiinterview.platform.application.resumeoptimization.dto.ResumeOptimizationCreatedResult;
import com.aiinterview.platform.application.resumeoptimization.dto.ResumeOptimizationInputFile;
import com.aiinterview.platform.application.resumeoptimization.port.ResumeOptimizationFileBindingPort;
import com.aiinterview.platform.application.resumeoptimization.port.ResumeOptimizationTaskCreation;
import com.aiinterview.platform.application.resumeoptimization.port.ResumeOptimizationTaskPort;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 简历优化应用服务。
 */
public class ResumeOptimizationApplicationService {

    private static final int VALIDATION_ERROR_CODE = 10001;
    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;
    private static final int DESCRIPTION_PREVIEW_MAX_LENGTH = 120;

    private final ResumeOptimizationFileBindingPort fileBindingPort;
    private final ResumeOptimizationTaskPort taskPort;
    private final ObjectMapper objectMapper;

    public ResumeOptimizationApplicationService(
            ResumeOptimizationFileBindingPort fileBindingPort,
            ResumeOptimizationTaskPort taskPort,
            ObjectMapper objectMapper
    ) {
        this.fileBindingPort = Objects.requireNonNull(fileBindingPort, "fileBindingPort 不能为空");
        this.taskPort = Objects.requireNonNull(taskPort, "taskPort 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    public ResumeOptimizationCreatedResult createOptimization(CreateResumeOptimizationCommand command) {
        Objects.requireNonNull(command, "command 不能为空");
        validate(command);
        ResumeOptimizationInputFile resumeFile = fileBindingPort.validateInputResume(command.userId(), command.resumeFileId());
        ResumeOptimizationInputFile avatarFile = command.avatarFileId() == null
                ? null
                : fileBindingPort.validateOptionalAvatar(command.userId(), command.avatarFileId());
        Map<String, Object> inputSummaryPayload = inputSummary(resumeFile, avatarFile, command);
        String inputSummary = toJson(inputSummaryPayload);
        Map<String, Object> workerPayload = workerPayload(resumeFile, avatarFile, command, inputSummaryPayload);
        return taskPort.createResumeOptimizationTask(new ResumeOptimizationTaskCreation(
                command.userId(),
                command.resumeFileId(),
                command.avatarFileId(),
                command.idempotencyKey(),
                command.requestHash(),
                command.traceId(),
                inputSummary,
                resumeFile,
                avatarFile,
                workerPayload
        ));
    }

    private static void validate(CreateResumeOptimizationCommand command) {
        if (command.targetJobTitle().isBlank() || command.targetJobTitle().length() > 80) {
            throw new ResumeOptimizationException(VALIDATION_ERROR_CODE, "目标岗位名称不合法");
        }
        if (command.targetJobDescription().isBlank()
                || command.targetJobDescription().length() < 20
                || command.targetJobDescription().length() > 4000) {
            throw new ResumeOptimizationException(VALIDATION_ERROR_CODE, "目标岗位描述不合法");
        }
    }

    private static Map<String, Object> inputSummary(
            ResumeOptimizationInputFile resumeFile,
            ResumeOptimizationInputFile avatarFile,
            CreateResumeOptimizationCommand command
    ) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("schema_version", 1);
        summary.put("resume_file_id", resumeFile.fileId());
        summary.put("resume_file_name", resumeFile.originalName());
        summary.put("avatar_file_id", avatarFile == null ? null : avatarFile.fileId());
        summary.put("avatar_present", avatarFile != null);
        summary.put("target_job_title", command.targetJobTitle());
        summary.put("target_job_description_preview", preview(command.targetJobDescription()));
        if (command.llmConfig() != null) {
            summary.put("llm_config", command.llmConfig().publicPayload());
        }
        return summary;
    }

    private static Map<String, Object> workerPayload(
            ResumeOptimizationInputFile resumeFile,
            ResumeOptimizationInputFile avatarFile,
            CreateResumeOptimizationCommand command,
            Map<String, Object> inputSummary
    ) {
        List<Map<String, Object>> inputFiles = new ArrayList<>();
        inputFiles.add(inputFilePayload(resumeFile));
        if (avatarFile != null) {
            inputFiles.add(inputFilePayload(avatarFile));
        }

        Map<String, Object> targetJob = new LinkedHashMap<>();
        targetJob.put("title", command.targetJobTitle());
        targetJob.put("description", command.targetJobDescription());

        Map<String, Object> optimizationOptions = new LinkedHashMap<>();
        optimizationOptions.put("prompt_version", "resume_optimization_v1");
        optimizationOptions.put("template", "default_pdf_v1");

        Map<String, Object> businessPayload = new LinkedHashMap<>();
        businessPayload.put("schema_version", 1);
        businessPayload.put("target_job", targetJob);
        businessPayload.put("optimization_options", optimizationOptions);
        if (command.llmConfig() != null) {
            businessPayload.put("llm_config", command.llmConfig().privatePayload());
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("user_id", command.userId());
        payload.put("input_files", inputFiles);
        payload.put("input_summary", inputSummary);
        payload.put("business_payload", businessPayload);
        return payload;
    }

    private static Map<String, Object> inputFilePayload(ResumeOptimizationInputFile inputFile) {
        Map<String, Object> inputFilePayload = new LinkedHashMap<>();
        inputFilePayload.put("file_id", inputFile.fileId());
        inputFilePayload.put("object_key", inputFile.objectKey());
        inputFilePayload.put("file_type", inputFile.fileType());
        inputFilePayload.put("file_purpose", inputFile.filePurpose());
        return inputFilePayload;
    }

    private String toJson(Map<String, Object> value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new ResumeOptimizationException(TASK_COMMAND_WRITE_FAILED_CODE, "任务输入摘要序列化失败", ex);
        }
    }

    private static String preview(String value) {
        if (value.length() <= DESCRIPTION_PREVIEW_MAX_LENGTH) {
            return value;
        }
        return value.substring(0, DESCRIPTION_PREVIEW_MAX_LENGTH);
    }
}
