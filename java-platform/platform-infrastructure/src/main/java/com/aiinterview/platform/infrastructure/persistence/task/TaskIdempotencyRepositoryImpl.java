package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.domain.task.DuplicateTaskIdempotencyException;
import com.aiinterview.platform.domain.task.TaskIdempotencyRecord;
import com.aiinterview.platform.domain.task.TaskIdempotencyRepository;
import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskIdempotencyEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.stereotype.Component;

/**
 * 任务幂等记录仓储基础设施实现。
 */
@Component
public class TaskIdempotencyRepositoryImpl implements TaskIdempotencyRepository {

    private final TaskIdempotencyMapper taskIdempotencyMapper;

    public TaskIdempotencyRepositoryImpl(TaskIdempotencyMapper taskIdempotencyMapper) {
        this.taskIdempotencyMapper = taskIdempotencyMapper;
    }

    @Override
    public void save(TaskIdempotencyRecord record) {
        try {
            taskIdempotencyMapper.insert(toEntity(record));
        } catch (DataIntegrityViolationException ex) {
            throw new DuplicateTaskIdempotencyException("任务幂等记录已存在");
        }
    }

    @Override
    public Optional<TaskIdempotencyRecord> findByUserIdAndOperationTypeAndIdempotencyKey(
            String userId,
            String operationType,
            String idempotencyKey
    ) {
        return Optional.ofNullable(taskIdempotencyMapper.selectOne(new LambdaQueryWrapper<TaskIdempotencyEntity>()
                .eq(TaskIdempotencyEntity::getUserId, userId)
                .eq(TaskIdempotencyEntity::getOperationType, operationType)
                .eq(TaskIdempotencyEntity::getIdempotencyKey, idempotencyKey)))
                .map(this::toDomain);
    }

    @Override
    public List<TaskIdempotencyRecord> findExpiredBefore(Instant expiresAt) {
        return taskIdempotencyMapper.selectList(new LambdaQueryWrapper<TaskIdempotencyEntity>()
                .lt(TaskIdempotencyEntity::getExpiresAt, expiresAt)
                .orderByAsc(TaskIdempotencyEntity::getExpiresAt))
                .stream().map(this::toDomain).toList();
    }

    private TaskIdempotencyRecord toDomain(TaskIdempotencyEntity entity) {
        return new TaskIdempotencyRecord(entity.getId(), entity.getUserId(), entity.getIdempotencyKey(),
                entity.getOperationType(), entity.getRequestHash(), entity.getTaskId(), entity.getResponseSnapshot(),
                entity.getExpiresAt(), entity.getCreatedAt(), entity.getUpdatedAt());
    }

    private TaskIdempotencyEntity toEntity(TaskIdempotencyRecord record) {
        TaskIdempotencyEntity entity = new TaskIdempotencyEntity();
        entity.setId(record.getId());
        entity.setUserId(record.getUserId());
        entity.setIdempotencyKey(record.getIdempotencyKey());
        entity.setOperationType(record.getOperationType());
        entity.setRequestHash(record.getRequestHash());
        entity.setTaskId(record.getTaskId());
        entity.setResponseSnapshot(record.getResponseSnapshot());
        entity.setExpiresAt(record.getExpiresAt());
        entity.setCreatedAt(record.getCreatedAt());
        entity.setUpdatedAt(record.getUpdatedAt());
        return entity;
    }
}
