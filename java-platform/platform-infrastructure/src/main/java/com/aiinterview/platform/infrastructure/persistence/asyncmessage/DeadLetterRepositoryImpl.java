package com.aiinterview.platform.infrastructure.persistence.asyncmessage;

import com.aiinterview.platform.domain.asyncmessage.DeadLetterRecord;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRepository;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterSourceType;
import com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity.DeadLetterEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * 死信记录仓储基础设施实现。
 */
@Component
public class DeadLetterRepositoryImpl implements DeadLetterRepository {

    private final DeadLetterMapper deadLetterMapper;

    public DeadLetterRepositoryImpl(DeadLetterMapper deadLetterMapper) {
        this.deadLetterMapper = deadLetterMapper;
    }

    @Override
    public void save(DeadLetterRecord record) {
        deadLetterMapper.insert(toEntity(record));
    }

    @Override
    public void update(DeadLetterRecord record) {
        deadLetterMapper.update(null, new LambdaUpdateWrapper<DeadLetterEntity>()
                .eq(DeadLetterEntity::getId, record.getId())
                .set(DeadLetterEntity::getManualHandled, record.isManualHandled())
                .set(DeadLetterEntity::getUpdatedAt, record.getUpdatedAt()));
    }

    @Override
    public Optional<DeadLetterRecord> findById(String id) {
        return Optional.ofNullable(deadLetterMapper.selectById(id)).map(this::toDomain);
    }

    @Override
    public Optional<DeadLetterRecord> findBySource(DeadLetterSourceType sourceType, String sourceId) {
        return Optional.ofNullable(deadLetterMapper.selectOne(new LambdaQueryWrapper<DeadLetterEntity>()
                .eq(DeadLetterEntity::getSourceType, sourceType.getCode())
                .eq(DeadLetterEntity::getSourceId, sourceId))).map(this::toDomain);
    }

    @Override
    public List<DeadLetterRecord> findByTaskId(String taskId) {
        return deadLetterMapper.selectList(new LambdaQueryWrapper<DeadLetterEntity>()
                        .eq(DeadLetterEntity::getTaskId, taskId)
                        .orderByDesc(DeadLetterEntity::getCreatedAt))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public List<DeadLetterRecord> findPage(DeadLetterSourceType sourceType, Boolean handled, int page, int pageSize) {
        return deadLetterMapper.selectPage(new Page<>(page, pageSize), query(sourceType, handled)
                        .orderByDesc(DeadLetterEntity::getCreatedAt))
                .getRecords()
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public long count(DeadLetterSourceType sourceType, Boolean handled) {
        return deadLetterMapper.selectCount(query(sourceType, handled));
    }

    private LambdaQueryWrapper<DeadLetterEntity> query(DeadLetterSourceType sourceType, Boolean handled) {
        LambdaQueryWrapper<DeadLetterEntity> wrapper = new LambdaQueryWrapper<>();
        if (sourceType != null) {
            wrapper.eq(DeadLetterEntity::getSourceType, sourceType.getCode());
        }
        if (handled != null) {
            wrapper.eq(DeadLetterEntity::getManualHandled, handled);
        }
        return wrapper;
    }

    private DeadLetterRecord toDomain(DeadLetterEntity entity) {
        return new DeadLetterRecord(
                entity.getId(),
                DeadLetterSourceType.fromCode(entity.getSourceType()).orElseThrow(
                        () -> new IllegalStateException("未知死信来源类型: " + entity.getSourceType())),
                entity.getSourceId(),
                entity.getTaskId(),
                entity.getTaskType(),
                entity.getAttemptNo(),
                entity.getFailureStage(),
                entity.getFailureReason(),
                entity.getRetryable(),
                entity.getManualHandled(),
                entity.getTraceId(),
                entity.getCreatedAt(),
                entity.getUpdatedAt()
        );
    }

    private static DeadLetterEntity toEntity(DeadLetterRecord record) {
        DeadLetterEntity entity = new DeadLetterEntity();
        entity.setId(record.getId());
        entity.setSourceType(record.getSourceType().getCode());
        entity.setSourceId(record.getSourceId());
        entity.setTaskId(record.getTaskId());
        entity.setTaskType(record.getTaskType());
        entity.setAttemptNo(record.getAttemptNo());
        entity.setFailureStage(record.getFailureStage());
        entity.setFailureReason(record.getFailureReason());
        entity.setRetryable(record.isRetryable());
        entity.setManualHandled(record.isManualHandled());
        entity.setTraceId(record.getTraceId());
        entity.setCreatedAt(record.getCreatedAt());
        entity.setUpdatedAt(record.getUpdatedAt());
        return entity;
    }
}
