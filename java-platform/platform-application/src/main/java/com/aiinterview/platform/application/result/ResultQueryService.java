package com.aiinterview.platform.application.result;

import com.aiinterview.platform.application.result.dto.ResultDetailResult;
import com.aiinterview.platform.application.result.dto.ResultFileItem;
import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FileBindType;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.result.TaskResult;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.aiinterview.platform.domain.task.TaskStatus;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 结果查询应用服务。
 */
public class ResultQueryService {

    private final TaskResultRepository taskResultRepository;
    private final FileBindingRepository fileBindingRepository;
    private final FileRepository fileRepository;

    public ResultQueryService(
            TaskResultRepository taskResultRepository,
            FileBindingRepository fileBindingRepository,
            FileRepository fileRepository
    ) {
        this.taskResultRepository = Objects.requireNonNull(taskResultRepository, "taskResultRepository 不能为空");
        this.fileBindingRepository = Objects.requireNonNull(fileBindingRepository, "fileBindingRepository 不能为空");
        this.fileRepository = Objects.requireNonNull(fileRepository, "fileRepository 不能为空");
    }

    public ResultDetailResult attachResultToTaskDetail(String taskId, String resultReferenceId, String taskStatus) {
        return attachResultToTaskDetail(taskId, null, resultReferenceId, taskStatus);
    }

    public ResultDetailResult attachResultToTaskDetail(
            String taskId,
            String taskType,
            String resultReferenceId,
            String taskStatus
    ) {
        if ("algorithm_test".equals(taskType)) {
            return attachAlgorithmTestResult(taskId, taskStatus);
        }
        if (!TaskStatus.SUCCESS.getCode().equals(taskStatus) || resultReferenceId == null || resultReferenceId.isBlank()) {
            return null;
        }
        return taskResultRepository.findById(resultReferenceId)
                .filter(result -> result.getTaskId().equals(taskId))
                .map(this::toDetail)
                .orElse(null);
    }

    private ResultDetailResult attachAlgorithmTestResult(String taskId, String taskStatus) {
        List<TaskResult> results = taskResultRepository.findByTaskId(taskId).stream()
                .filter(result -> "algorithm_test".equals(result.getResultType().getCode()))
                .toList();
        if (TaskStatus.WAITING_USER_INPUT.getCode().equals(taskStatus)) {
            return results.stream()
                    .filter(result -> "problem_set".equals(result.getSummary().get("phase")))
                    .min(Comparator.comparingInt(TaskResult::getAttemptNo))
                    .map(result -> toDetail(result, Map.of("check_available", true)))
                    .orElse(null);
        }
        if (TaskStatus.SUCCESS.getCode().equals(taskStatus)) {
            return results.stream()
                    .filter(result -> "check_result".equals(result.getSummary().get("phase")))
                    .max(Comparator.comparingInt(TaskResult::getAttemptNo))
                    .map(result -> toDetail(result, Map.of()))
                    .orElse(null);
        }
        return null;
    }

    private ResultDetailResult toDetail(TaskResult result) {
        return toDetail(result, Map.of());
    }

    private ResultDetailResult toDetail(TaskResult result, Map<String, Object> summaryExtensions) {
        Map<String, Object> summary = new LinkedHashMap<>(result.getSummary());
        summary.putAll(summaryExtensions);
        return new ResultDetailResult(
                result.getId(),
                result.getResultType().getCode(),
                summary,
                files(result.getTaskId(), result.getAttemptNo(), FileBindType.REPORT),
                files(result.getTaskId(), result.getAttemptNo(), FileBindType.OUTPUT),
                result.getResultStatus().getCode(),
                result.getCreatedAt(),
                result.getCompletedAt()
        );
    }

    private List<ResultFileItem> files(String taskId, int attemptNo, FileBindType bindType) {
        return fileBindingRepository.findByTaskIdAndBindType(taskId, bindType)
                .stream()
                .filter(binding -> binding.getAttemptNo() == attemptNo)
                .map(binding -> fileRepository.findById(binding.getFileId()).orElse(null))
                .filter(Objects::nonNull)
                .map(this::toFileItem)
                .toList();
    }

    private ResultFileItem toFileItem(File file) {
        return new ResultFileItem(
                file.getId(),
                file.getOriginalName(),
                file.getFileType().getCode(),
                file.getFileSize()
        );
    }
}
