package com.aiinterview.platform.application.resumeevaluation;

import com.aiinterview.platform.application.resumeevaluation.command.CreateResumeEvaluationCommand;
import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationCreatedResult;
import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationInputFile;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationFileBindingPort;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationTaskCreation;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationTaskPort;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 简历评估应用服务。
 */
public class ResumeEvaluationApplicationService {

    private static final int VALIDATION_ERROR_CODE = 10001;
    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;
    private static final int DESCRIPTION_PREVIEW_MAX_LENGTH = 120;

    private final ResumeEvaluationFileBindingPort fileBindingPort;
    private final ResumeEvaluationTaskPort taskPort;
    private final ObjectMapper objectMapper;

    public ResumeEvaluationApplicationService(
            ResumeEvaluationFileBindingPort fileBindingPort,
            ResumeEvaluationTaskPort taskPort,
            ObjectMapper objectMapper
    ) {
        this.fileBindingPort = Objects.requireNonNull(fileBindingPort, "fileBindingPort 不能为空");
        this.taskPort = Objects.requireNonNull(taskPort, "taskPort 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    public ResumeEvaluationCreatedResult createEvaluation(CreateResumeEvaluationCommand command) {
        Objects.requireNonNull(command, "command 不能为空");
        validate(command);
        ResumeEvaluationInputFile inputFile = fileBindingPort.validateInputResume(command.userId(), command.resumeFileId());
        Map<String, Object> inputSummaryPayload = inputSummary(inputFile, command);
        String inputSummary = toJson(inputSummaryPayload);
        Map<String, Object> workerPayload = workerPayload(inputFile, command, inputSummaryPayload);
        return taskPort.createResumeEvaluationTask(new ResumeEvaluationTaskCreation(
                command.userId(),
                command.resumeFileId(),
                command.idempotencyKey(),
                command.requestHash(),
                command.traceId(),
                inputSummary,
                inputFile,
                workerPayload
        ));
    }

    private static void validate(CreateResumeEvaluationCommand command) {
        if (command.targetJobTitle().isBlank() || command.targetJobTitle().length() > 80) {
            throw new ResumeEvaluationException(VALIDATION_ERROR_CODE, "目标岗位名称不合法");
        }
        if (command.targetJobDescription().isBlank()
                || command.targetJobDescription().length() < 20
                || command.targetJobDescription().length() > 4000) {
            throw new ResumeEvaluationException(VALIDATION_ERROR_CODE, "目标岗位描述不合法");
        }
    }

    private static Map<String, Object> inputSummary(ResumeEvaluationInputFile inputFile, CreateResumeEvaluationCommand command) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("schema_version", 1);
        summary.put("resume_file_id", inputFile.fileId());
        summary.put("resume_file_name", inputFile.originalName());
        summary.put("target_job_title", command.targetJobTitle());
        summary.put("target_job_description_preview", preview(command.targetJobDescription()));
        if (command.llmConfig() != null) {
            summary.put("llm_config", command.llmConfig().publicPayload());
        }
        return summary;
    }

    private static Map<String, Object> workerPayload(
            ResumeEvaluationInputFile inputFile,
            CreateResumeEvaluationCommand command,
            Map<String, Object> inputSummary
    ) {
        Map<String, Object> inputFilePayload = new LinkedHashMap<>();
        inputFilePayload.put("file_id", inputFile.fileId());
        inputFilePayload.put("object_key", inputFile.objectKey());
        inputFilePayload.put("file_type", inputFile.fileType());
        inputFilePayload.put("file_purpose", inputFile.filePurpose());

        Map<String, Object> targetJob = new LinkedHashMap<>();
        targetJob.put("title", command.targetJobTitle());
        targetJob.put("description", command.targetJobDescription());

        Map<String, Object> evaluationOptions = new LinkedHashMap<>();
        evaluationOptions.put("dimension_profile", "default_v1");
        evaluationOptions.put("language", "zh-CN");

        Map<String, Object> businessPayload = new LinkedHashMap<>();
        businessPayload.put("schema_version", 1);
        businessPayload.put("target_job", targetJob);
        businessPayload.put("evaluation_options", evaluationOptions);
        if (command.llmConfig() != null) {
            businessPayload.put("llm_config", command.llmConfig().privatePayload());
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("user_id", command.userId());
        payload.put("input_files", List.of(inputFilePayload));
        payload.put("input_summary", inputSummary);
        payload.put("business_payload", businessPayload);
        return payload;
    }

    private String toJson(Map<String, Object> value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new ResumeEvaluationException(TASK_COMMAND_WRITE_FAILED_CODE, "任务输入摘要序列化失败", ex);
        }
    }

    private static String preview(String value) {
        if (value.length() <= DESCRIPTION_PREVIEW_MAX_LENGTH) {
            return value;
        }
        return value.substring(0, DESCRIPTION_PREVIEW_MAX_LENGTH);
    }
}
