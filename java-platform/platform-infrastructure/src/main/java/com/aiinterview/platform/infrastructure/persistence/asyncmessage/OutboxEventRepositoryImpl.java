package com.aiinterview.platform.infrastructure.persistence.asyncmessage;

import com.aiinterview.platform.domain.asyncmessage.AsyncDeliveryStatus;
import com.aiinterview.platform.domain.asyncmessage.OutboxEvent;
import com.aiinterview.platform.domain.asyncmessage.OutboxEventRepository;
import com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity.OutboxEventEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * Outbox 事件仓储基础设施实现。
 */
@Component
public class OutboxEventRepositoryImpl implements OutboxEventRepository {

    private final OutboxEventMapper outboxEventMapper;

    public OutboxEventRepositoryImpl(OutboxEventMapper outboxEventMapper) {
        this.outboxEventMapper = outboxEventMapper;
    }

    @Override
    public void save(OutboxEvent event) {
        outboxEventMapper.insert(toEntity(event));
    }

    @Override
    public void update(OutboxEvent event) {
        outboxEventMapper.update(null, updateWrapper(event));
    }

    @Override
    public boolean updateByIdAndStatus(OutboxEvent event, AsyncDeliveryStatus expectedStatus) {
        int rows = outboxEventMapper.update(null, updateWrapper(event)
                .eq(OutboxEventEntity::getStatus, expectedStatus.getCode()));
        return rows > 0;
    }

    @Override
    public Optional<OutboxEvent> findById(String id) {
        return Optional.ofNullable(outboxEventMapper.selectById(id)).map(this::toDomain);
    }

    @Override
    public Optional<OutboxEvent> findByEventId(String eventId) {
        return Optional.ofNullable(outboxEventMapper.selectOne(new LambdaQueryWrapper<OutboxEventEntity>()
                .eq(OutboxEventEntity::getEventId, eventId))).map(this::toDomain);
    }

    @Override
    public Optional<OutboxEvent> findByTaskIdAndAttemptNoAndEventType(String taskId, int attemptNo, String eventType) {
        return Optional.ofNullable(outboxEventMapper.selectOne(new LambdaQueryWrapper<OutboxEventEntity>()
                .eq(OutboxEventEntity::getTaskId, taskId)
                .eq(OutboxEventEntity::getAttemptNo, attemptNo)
                .eq(OutboxEventEntity::getEventType, eventType)
                .orderByDesc(OutboxEventEntity::getCreatedAt)
                .last("LIMIT 1"))).map(this::toDomain);
    }

    @Override
    public List<OutboxEvent> findByTaskId(String taskId) {
        return outboxEventMapper.selectList(new LambdaQueryWrapper<OutboxEventEntity>()
                        .eq(OutboxEventEntity::getTaskId, taskId)
                        .orderByDesc(OutboxEventEntity::getCreatedAt))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public List<OutboxEvent> findPublishable(Instant now, int limit) {
        return outboxEventMapper.selectList(new LambdaQueryWrapper<OutboxEventEntity>()
                        .and(wrapper -> wrapper
                                .eq(OutboxEventEntity::getStatus, AsyncDeliveryStatus.PENDING.getCode())
                                .or(retryable -> retryable
                                        .eq(OutboxEventEntity::getStatus, AsyncDeliveryStatus.FAILED.getCode())
                                        .apply("retry_count < max_retry_count")
                                        .and(nextRetry -> nextRetry
                                                .isNull(OutboxEventEntity::getNextRetryAt)
                                                .or()
                                                .le(OutboxEventEntity::getNextRetryAt, now))))
                        .orderByAsc(OutboxEventEntity::getCreatedAt)
                        .last("LIMIT " + Math.max(limit, 0)))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public List<OutboxEvent> findStaleDelivering(Instant before, int limit) {
        return outboxEventMapper.selectList(new LambdaQueryWrapper<OutboxEventEntity>()
                        .eq(OutboxEventEntity::getStatus, AsyncDeliveryStatus.DELIVERING.getCode())
                        .lt(OutboxEventEntity::getUpdatedAt, before)
                        .orderByAsc(OutboxEventEntity::getUpdatedAt)
                        .last("LIMIT " + Math.max(limit, 0)))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    private LambdaUpdateWrapper<OutboxEventEntity> updateWrapper(OutboxEvent event) {
        return new LambdaUpdateWrapper<OutboxEventEntity>()
                .eq(OutboxEventEntity::getId, event.getId())
                .set(OutboxEventEntity::getEventId, event.getEventId())
                .set(OutboxEventEntity::getAggregateType, event.getAggregateType())
                .set(OutboxEventEntity::getAggregateId, event.getAggregateId())
                .set(OutboxEventEntity::getTaskId, event.getTaskId())
                .set(OutboxEventEntity::getTaskType, event.getTaskType())
                .set(OutboxEventEntity::getAttemptNo, event.getAttemptNo())
                .set(OutboxEventEntity::getEventType, event.getEventType())
                .set(OutboxEventEntity::getRoutingKey, event.getRoutingKey())
                .set(OutboxEventEntity::getPayload, event.getPayload())
                .set(OutboxEventEntity::getStatus, event.getStatus().getCode())
                .set(OutboxEventEntity::getRetryCount, event.getRetryCount())
                .set(OutboxEventEntity::getMaxRetryCount, event.getMaxRetryCount())
                .set(OutboxEventEntity::getNextRetryAt, event.getNextRetryAt())
                .set(OutboxEventEntity::getLastError, event.getLastError())
                .set(OutboxEventEntity::getTraceId, event.getTraceId())
                .set(OutboxEventEntity::getCreatedAt, event.getCreatedAt())
                .set(OutboxEventEntity::getUpdatedAt, event.getUpdatedAt())
                .set(OutboxEventEntity::getDeliveredAt, event.getDeliveredAt());
    }

    private OutboxEvent toDomain(OutboxEventEntity entity) {
        return new OutboxEvent(
                entity.getId(),
                entity.getEventId(),
                entity.getAggregateType(),
                entity.getAggregateId(),
                entity.getTaskId(),
                entity.getTaskType(),
                entity.getAttemptNo(),
                entity.getEventType(),
                entity.getRoutingKey(),
                entity.getPayload(),
                AsyncDeliveryStatus.fromCode(entity.getStatus()).orElseThrow(
                        () -> new IllegalStateException("未知 Outbox 状态: " + entity.getStatus())),
                entity.getRetryCount(),
                entity.getMaxRetryCount(),
                entity.getNextRetryAt(),
                entity.getLastError(),
                entity.getTraceId(),
                entity.getCreatedAt(),
                entity.getUpdatedAt(),
                entity.getDeliveredAt()
        );
    }

    private static OutboxEventEntity toEntity(OutboxEvent event) {
        OutboxEventEntity entity = new OutboxEventEntity();
        entity.setId(event.getId());
        entity.setEventId(event.getEventId());
        entity.setAggregateType(event.getAggregateType());
        entity.setAggregateId(event.getAggregateId());
        entity.setTaskId(event.getTaskId());
        entity.setTaskType(event.getTaskType());
        entity.setAttemptNo(event.getAttemptNo());
        entity.setEventType(event.getEventType());
        entity.setRoutingKey(event.getRoutingKey());
        entity.setPayload(event.getPayload());
        entity.setStatus(event.getStatus().getCode());
        entity.setRetryCount(event.getRetryCount());
        entity.setMaxRetryCount(event.getMaxRetryCount());
        entity.setNextRetryAt(event.getNextRetryAt());
        entity.setLastError(event.getLastError());
        entity.setTraceId(event.getTraceId());
        entity.setCreatedAt(event.getCreatedAt());
        entity.setUpdatedAt(event.getUpdatedAt());
        entity.setDeliveredAt(event.getDeliveredAt());
        return entity;
    }
}
