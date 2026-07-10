package com.aiinterview.platform.infrastructure.persistence.result;

import com.aiinterview.platform.domain.result.ResultType;
import com.aiinterview.platform.domain.result.TaskResult;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.aiinterview.platform.infrastructure.persistence.result.entity.TaskResultEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * 任务结果仓储基础设施实现。
 */
@Component
public class TaskResultRepositoryImpl implements TaskResultRepository {

    private static final TypeReference<Map<String, Object>> SUMMARY_TYPE = new TypeReference<>() {
    };

    private final TaskResultMapper taskResultMapper;
    private final ObjectMapper objectMapper;

    public TaskResultRepositoryImpl(TaskResultMapper taskResultMapper, ObjectMapper objectMapper) {
        this.taskResultMapper = taskResultMapper;
        this.objectMapper = objectMapper;
    }

    @Override
    public void save(TaskResult result) {
        taskResultMapper.insert(toEntity(result));
    }

    @Override
    public Optional<TaskResult> findById(String resultId) {
        return Optional.ofNullable(taskResultMapper.selectById(resultId)).map(this::toDomain);
    }

    @Override
    public Optional<TaskResult> findByTaskIdAndAttemptNo(String taskId, int attemptNo) {
        return taskResultMapper.selectList(new LambdaQueryWrapper<TaskResultEntity>()
                        .eq(TaskResultEntity::getTaskId, taskId)
                        .eq(TaskResultEntity::getAttemptNo, attemptNo))
                .stream()
                .findFirst()
                .map(this::toDomain);
    }

    @Override
    public List<TaskResult> findByTaskId(String taskId) {
        return taskResultMapper.selectList(new LambdaQueryWrapper<TaskResultEntity>()
                        .eq(TaskResultEntity::getTaskId, taskId)
                        .orderByDesc(TaskResultEntity::getCreatedAt))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    private TaskResult toDomain(TaskResultEntity entity) {
        return TaskResult.createCompleted(
                entity.getId(),
                entity.getTaskId(),
                entity.getAttemptNo(),
                ResultType.fromCode(entity.getResultType()).orElseThrow(),
                parseSummary(entity.getSummary()),
                entity.getCompletedAt()
        );
    }

    private TaskResultEntity toEntity(TaskResult result) {
        TaskResultEntity entity = new TaskResultEntity();
        entity.setId(result.getId());
        entity.setTaskId(result.getTaskId());
        entity.setAttemptNo(result.getAttemptNo());
        entity.setResultType(result.getResultType().getCode());
        entity.setSummary(writeSummary(result.getSummary()));
        entity.setResultStatus(result.getResultStatus().getCode());
        entity.setCreatedAt(result.getCreatedAt());
        entity.setUpdatedAt(result.getUpdatedAt());
        entity.setCompletedAt(result.getCompletedAt());
        return entity;
    }

    private Map<String, Object> parseSummary(String summary) {
        try {
            return objectMapper.readValue(summary, SUMMARY_TYPE);
        } catch (Exception ex) {
            throw new IllegalStateException("结果摘要反序列化失败", ex);
        }
    }

    private String writeSummary(Map<String, Object> summary) {
        try {
            return objectMapper.writeValueAsString(summary);
        } catch (Exception ex) {
            throw new IllegalStateException("结果摘要序列化失败", ex);
        }
    }
}
