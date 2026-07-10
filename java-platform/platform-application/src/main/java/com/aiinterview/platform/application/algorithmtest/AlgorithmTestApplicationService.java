package com.aiinterview.platform.application.algorithmtest;

import com.aiinterview.platform.application.algorithmtest.command.CreateAlgorithmTestCommand;
import com.aiinterview.platform.application.algorithmtest.command.RequestAlgorithmCheckCommand;
import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmCheckRequestedResult;
import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmTestCreatedResult;
import com.aiinterview.platform.application.algorithmtest.port.AlgorithmTestCheckRequest;
import com.aiinterview.platform.application.algorithmtest.port.AlgorithmTestTaskCreation;
import com.aiinterview.platform.application.algorithmtest.port.AlgorithmTestTaskPort;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 算法测试应用服务。
 */
public class AlgorithmTestApplicationService {

    public static final int PROBLEM_COUNT = 5;
    public static final int PASS_THRESHOLD = 2;

    private static final int TASK_COMMAND_WRITE_FAILED_CODE = 71014;

    private final AlgorithmTestTaskPort taskPort;
    private final ObjectMapper objectMapper;

    public AlgorithmTestApplicationService(AlgorithmTestTaskPort taskPort, ObjectMapper objectMapper) {
        this.taskPort = Objects.requireNonNull(taskPort, "taskPort 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    public AlgorithmTestCreatedResult createAlgorithmTest(CreateAlgorithmTestCommand command) {
        Objects.requireNonNull(command, "command 不能为空");
        Map<String, Object> inputSummary = inputSummary(command);
        String inputSummaryJson = toJson(inputSummary);
        return taskPort.createAlgorithmTestTask(new AlgorithmTestTaskCreation(
                command.userId(),
                command.idempotencyKey(),
                command.requestHash(),
                command.traceId(),
                inputSummaryJson,
                workerPayload(command, inputSummary)
        ));
    }

    public AlgorithmCheckRequestedResult requestCheck(RequestAlgorithmCheckCommand command) {
        Objects.requireNonNull(command, "command 不能为空");
        return taskPort.requestAlgorithmCheck(new AlgorithmTestCheckRequest(
                command.userId(),
                command.taskId(),
                command.idempotencyKey(),
                command.requestHash(),
                command.traceId()
        ));
    }

    private static Map<String, Object> inputSummary(CreateAlgorithmTestCommand command) {
        Map<String, Object> targetJob = new LinkedHashMap<>();
        targetJob.put("title", command.targetJobTitle());
        targetJob.put("description", command.targetJobDescription());

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("schema_version", 1);
        summary.put("action", "generate_problem_set");
        summary.put("target_job", targetJob);
        summary.put("codeforces_handle", command.codeforcesHandle());
        summary.put("problem_count", PROBLEM_COUNT);
        summary.put("pass_threshold", PASS_THRESHOLD);
        if (command.llmConfig() != null) {
            summary.put("llm_config", command.llmConfig().publicPayload());
        }
        return summary;
    }

    private static Map<String, Object> workerPayload(
            CreateAlgorithmTestCommand command,
            Map<String, Object> inputSummary
    ) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("task_type", "algorithm_test");
        payload.put("user_id", command.userId());
        payload.put("input_summary", inputSummary);
        payload.put("input_files", List.of());
        if (command.llmConfig() != null) {
            payload.put("llm_config", command.llmConfig().privatePayload());
        }
        return payload;
    }

    private String toJson(Map<String, Object> value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new AlgorithmTestException(TASK_COMMAND_WRITE_FAILED_CODE, "任务输入摘要序列化失败", ex);
        }
    }
}
