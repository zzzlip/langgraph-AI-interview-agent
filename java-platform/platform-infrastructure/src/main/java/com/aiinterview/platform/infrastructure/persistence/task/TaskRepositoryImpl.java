package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.domain.task.FailedTaskQueryCriteria;
import com.aiinterview.platform.domain.task.PageResult;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskQueryCriteria;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * 任务仓储基础设施实现。
 */
@Component
public class TaskRepositoryImpl implements TaskRepository {

    private final TaskMapper taskMapper;

    public TaskRepositoryImpl(TaskMapper taskMapper) {
        this.taskMapper = taskMapper;
    }

    @Override
    public void save(Task task) {
        taskMapper.insert(toEntity(task));
    }

    @Override
    public Optional<Task> findById(String taskId) {
        return Optional.ofNullable(taskMapper.selectById(taskId)).map(this::toDomain);
    }

    @Override
    public Optional<Task> findByTraceId(String traceId) {
        return Optional.ofNullable(taskMapper.selectOne(new LambdaQueryWrapper<TaskEntity>()
                .eq(TaskEntity::getTraceId, traceId))).map(this::toDomain);
    }

    @Override
    public PageResult<Task> findByUser(TaskQueryCriteria criteria) {
        LambdaQueryWrapper<TaskEntity> wrapper = new LambdaQueryWrapper<TaskEntity>()
                .eq(TaskEntity::getUserId, criteria.userId())
                .eq(criteria.taskType() != null, TaskEntity::getTaskType, criteria.taskType() == null ? null : criteria.taskType().getCode())
                .eq(criteria.status() != null, TaskEntity::getStatus, criteria.status() == null ? null : criteria.status().getCode())
                .orderByDesc(TaskEntity::getCreatedAt);
        long total = taskMapper.selectCount(wrapper);
        int offset = Math.max(criteria.page() - 1, 0) * criteria.pageSize();
        wrapper.last("LIMIT " + criteria.pageSize() + " OFFSET " + offset);
        List<Task> items = taskMapper.selectList(wrapper).stream().map(this::toDomain).toList();
        return new PageResult<>(items, total, criteria.page(), criteria.pageSize());
    }

    @Override
    public PageResult<Task> findFailed(FailedTaskQueryCriteria criteria) {
        LambdaQueryWrapper<TaskEntity> wrapper = new LambdaQueryWrapper<TaskEntity>()
                .eq(TaskEntity::getStatus, "failed")
                .eq(criteria.taskType() != null, TaskEntity::getTaskType, criteria.taskType() == null ? null : criteria.taskType().getCode())
                .eq(criteria.retryable() != null, TaskEntity::getRetryable, criteria.retryable())
                .orderByDesc(TaskEntity::getLastFailedAt);
        long total = taskMapper.selectCount(wrapper);
        int offset = Math.max(criteria.page() - 1, 0) * criteria.pageSize();
        wrapper.last("LIMIT " + criteria.pageSize() + " OFFSET " + offset);
        List<Task> items = taskMapper.selectList(wrapper).stream().map(this::toDomain).toList();
        return new PageResult<>(items, total, criteria.page(), criteria.pageSize());
    }

    @Override
    public boolean updateByIdAndVersion(Task task, long expectedVersion) {
        TaskEntity entity = toEntity(task);
        entity.setVersion(expectedVersion + 1);
        int rows = taskMapper.update(entity, new LambdaUpdateWrapper<TaskEntity>()
                .eq(TaskEntity::getId, task.getId())
                .eq(TaskEntity::getVersion, expectedVersion));
        return rows > 0;
    }

    private Task toDomain(TaskEntity entity) {
        return new Task(
                entity.getId(),
                entity.getUserId(),
                TaskEnumMapper.typeFromCode(entity.getTaskType()),
                TaskEnumMapper.statusFromCode(entity.getStatus()),
                entity.getStage(),
                entity.getProgress(),
                entity.getInputSummary(),
                entity.getResultReferenceId(),
                entity.getCurrentAttempt(),
                entity.getRetryCount(),
                entity.getMaxRetryCount(),
                Boolean.TRUE.equals(entity.getRetryable()),
                entity.getVersion(),
                entity.getTraceId(),
                entity.getCreatedAt(),
                entity.getUpdatedAt(),
                entity.getCompletedAt(),
                entity.getCancelledAt(),
                entity.getLastFailedAt()
        );
    }

    private TaskEntity toEntity(Task task) {
        TaskEntity entity = new TaskEntity();
        entity.setId(task.getId());
        entity.setUserId(task.getUserId());
        entity.setTaskType(task.getTaskType().getCode());
        entity.setStatus(task.getStatus().getCode());
        entity.setStage(task.getStage());
        entity.setProgress(task.getProgress());
        entity.setInputSummary(task.getInputSummary());
        entity.setResultReferenceId(task.getResultReferenceId());
        entity.setCurrentAttempt(task.getCurrentAttempt());
        entity.setRetryCount(task.getRetryCount());
        entity.setMaxRetryCount(task.getMaxRetryCount());
        entity.setRetryable(task.isRetryable());
        entity.setVersion(task.getVersion());
        entity.setTraceId(task.getTraceId());
        entity.setCreatedAt(task.getCreatedAt());
        entity.setUpdatedAt(task.getUpdatedAt());
        entity.setCompletedAt(task.getCompletedAt());
        entity.setCancelledAt(task.getCancelledAt());
        entity.setLastFailedAt(task.getLastFailedAt());
        return entity;
    }
}
