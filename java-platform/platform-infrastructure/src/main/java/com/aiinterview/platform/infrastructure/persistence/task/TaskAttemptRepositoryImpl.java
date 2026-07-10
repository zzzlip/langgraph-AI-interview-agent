package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.domain.task.TaskAttempt;
import com.aiinterview.platform.domain.task.TaskAttemptRepository;
import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskAttemptEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * 任务执行尝试仓储基础设施实现。
 */
@Component
public class TaskAttemptRepositoryImpl implements TaskAttemptRepository {

    private final TaskAttemptMapper taskAttemptMapper;

    public TaskAttemptRepositoryImpl(TaskAttemptMapper taskAttemptMapper) {
        this.taskAttemptMapper = taskAttemptMapper;
    }

    @Override
    public void save(TaskAttempt attempt) {
        taskAttemptMapper.insert(toEntity(attempt));
    }

    @Override
    public Optional<TaskAttempt> findById(String attemptId) {
        return Optional.ofNullable(taskAttemptMapper.selectById(attemptId)).map(this::toDomain);
    }

    @Override
    public Optional<TaskAttempt> findByTaskIdAndAttemptNo(String taskId, int attemptNo) {
        return Optional.ofNullable(taskAttemptMapper.selectOne(new LambdaQueryWrapper<TaskAttemptEntity>()
                .eq(TaskAttemptEntity::getTaskId, taskId)
                .eq(TaskAttemptEntity::getAttemptNo, attemptNo))).map(this::toDomain);
    }

    @Override
    public List<TaskAttempt> findByTaskId(String taskId) {
        return taskAttemptMapper.selectList(new LambdaQueryWrapper<TaskAttemptEntity>()
                .eq(TaskAttemptEntity::getTaskId, taskId)
                .orderByAsc(TaskAttemptEntity::getAttemptNo))
                .stream().map(this::toDomain).toList();
    }

    @Override
    public void update(TaskAttempt attempt) {
        taskAttemptMapper.updateById(toEntity(attempt));
    }

    private TaskAttempt toDomain(TaskAttemptEntity entity) {
        return new TaskAttempt(entity.getId(), entity.getTaskId(), entity.getAttemptNo(),
                TaskEnumMapper.statusFromCode(entity.getStatus()), entity.getStartedAt(), entity.getFinishedAt(),
                entity.getFailedAt(), entity.getFailureStage(), entity.getFailureCode(),
                entity.getUserMessage(), entity.getAdminSummary(), entity.getCreatedAt(), entity.getUpdatedAt());
    }

    private TaskAttemptEntity toEntity(TaskAttempt attempt) {
        TaskAttemptEntity entity = new TaskAttemptEntity();
        entity.setId(attempt.getId());
        entity.setTaskId(attempt.getTaskId());
        entity.setAttemptNo(attempt.getAttemptNo());
        entity.setStatus(attempt.getStatus().getCode());
        entity.setStartedAt(attempt.getStartedAt());
        entity.setFinishedAt(attempt.getFinishedAt());
        entity.setFailedAt(attempt.getFailedAt());
        entity.setFailureStage(attempt.getFailureStage());
        entity.setFailureCode(attempt.getFailureCode());
        entity.setUserMessage(attempt.getUserMessage());
        entity.setAdminSummary(attempt.getAdminSummary());
        entity.setCreatedAt(attempt.getCreatedAt());
        entity.setUpdatedAt(attempt.getUpdatedAt());
        return entity;
    }
}
