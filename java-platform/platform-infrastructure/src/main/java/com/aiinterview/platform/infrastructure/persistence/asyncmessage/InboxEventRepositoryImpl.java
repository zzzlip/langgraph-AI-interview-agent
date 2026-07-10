package com.aiinterview.platform.infrastructure.persistence.asyncmessage;

import com.aiinterview.platform.domain.asyncmessage.AsyncConsumptionStatus;
import com.aiinterview.platform.domain.asyncmessage.InboxEvent;
import com.aiinterview.platform.domain.asyncmessage.InboxEventRepository;
import com.aiinterview.platform.domain.asyncmessage.WorkerFeedbackType;
import com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity.InboxEventEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * Inbox 事件仓储基础设施实现。
 */
@Component
public class InboxEventRepositoryImpl implements InboxEventRepository {

    private final InboxEventMapper inboxEventMapper;

    public InboxEventRepositoryImpl(InboxEventMapper inboxEventMapper) {
        this.inboxEventMapper = inboxEventMapper;
    }

    @Override
    public void save(InboxEvent event) {
        inboxEventMapper.insert(toEntity(event));
    }

    @Override
    public void update(InboxEvent event) {
        inboxEventMapper.update(null, updateWrapper(event));
    }

    @Override
    public Optional<InboxEvent> findById(String id) {
        return Optional.ofNullable(inboxEventMapper.selectById(id)).map(this::toDomain);
    }

    @Override
    public Optional<InboxEvent> findByEventId(String eventId) {
        return Optional.ofNullable(inboxEventMapper.selectOne(new LambdaQueryWrapper<InboxEventEntity>()
                .eq(InboxEventEntity::getEventId, eventId))).map(this::toDomain);
    }

    @Override
    public List<InboxEvent> findByTaskIdAndAttemptNo(String taskId, int attemptNo) {
        return inboxEventMapper.selectList(new LambdaQueryWrapper<InboxEventEntity>()
                        .eq(InboxEventEntity::getTaskId, taskId)
                        .eq(InboxEventEntity::getAttemptNo, attemptNo)
                        .orderByAsc(InboxEventEntity::getReceivedAt))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public Optional<InboxEvent> findConsumedByTaskIdAndAttemptNo(String taskId, int attemptNo) {
        return Optional.ofNullable(inboxEventMapper.selectOne(new LambdaQueryWrapper<InboxEventEntity>()
                .eq(InboxEventEntity::getTaskId, taskId)
                .eq(InboxEventEntity::getAttemptNo, attemptNo)
                .eq(InboxEventEntity::getFeedbackType, WorkerFeedbackType.CONSUMED.getCode())
                .last("LIMIT 1"))).map(this::toDomain);
    }

    private LambdaUpdateWrapper<InboxEventEntity> updateWrapper(InboxEvent event) {
        return new LambdaUpdateWrapper<InboxEventEntity>()
                .eq(InboxEventEntity::getId, event.getId())
                .set(InboxEventEntity::getEventId, event.getEventId())
                .set(InboxEventEntity::getEventType, event.getEventType())
                .set(InboxEventEntity::getTaskId, event.getTaskId())
                .set(InboxEventEntity::getTaskType, event.getTaskType())
                .set(InboxEventEntity::getAttemptNo, event.getAttemptNo())
                .set(InboxEventEntity::getFeedbackType, event.getFeedbackType().getCode())
                .set(InboxEventEntity::getStatus, event.getStatus().getCode())
                .set(InboxEventEntity::getPayloadHash, event.getPayloadHash())
                .set(InboxEventEntity::getPayloadSummary, event.getPayloadSummary())
                .set(InboxEventEntity::getRetryCount, event.getRetryCount())
                .set(InboxEventEntity::getMaxRetryCount, event.getMaxRetryCount())
                .set(InboxEventEntity::getLastError, event.getLastError())
                .set(InboxEventEntity::getTraceId, event.getTraceId())
                .set(InboxEventEntity::getReceivedAt, event.getReceivedAt())
                .set(InboxEventEntity::getProcessedAt, event.getProcessedAt())
                .set(InboxEventEntity::getCreatedAt, event.getCreatedAt())
                .set(InboxEventEntity::getUpdatedAt, event.getUpdatedAt());
    }

    private InboxEvent toDomain(InboxEventEntity entity) {
        return new InboxEvent(
                entity.getId(),
                entity.getEventId(),
                entity.getEventType(),
                entity.getTaskId(),
                entity.getTaskType(),
                entity.getAttemptNo(),
                WorkerFeedbackType.fromCode(entity.getFeedbackType()).orElseThrow(
                        () -> new IllegalStateException("未知 Worker 反馈类型: " + entity.getFeedbackType())),
                AsyncConsumptionStatus.fromCode(entity.getStatus()).orElseThrow(
                        () -> new IllegalStateException("未知 Inbox 状态: " + entity.getStatus())),
                entity.getPayloadHash(),
                entity.getPayloadSummary(),
                entity.getRetryCount(),
                entity.getMaxRetryCount(),
                entity.getLastError(),
                entity.getTraceId(),
                entity.getReceivedAt(),
                entity.getProcessedAt(),
                entity.getCreatedAt(),
                entity.getUpdatedAt()
        );
    }

    private static InboxEventEntity toEntity(InboxEvent event) {
        InboxEventEntity entity = new InboxEventEntity();
        entity.setId(event.getId());
        entity.setEventId(event.getEventId());
        entity.setEventType(event.getEventType());
        entity.setTaskId(event.getTaskId());
        entity.setTaskType(event.getTaskType());
        entity.setAttemptNo(event.getAttemptNo());
        entity.setFeedbackType(event.getFeedbackType().getCode());
        entity.setStatus(event.getStatus().getCode());
        entity.setPayloadHash(event.getPayloadHash());
        entity.setPayloadSummary(event.getPayloadSummary());
        entity.setRetryCount(event.getRetryCount());
        entity.setMaxRetryCount(event.getMaxRetryCount());
        entity.setLastError(event.getLastError());
        entity.setTraceId(event.getTraceId());
        entity.setReceivedAt(event.getReceivedAt());
        entity.setProcessedAt(event.getProcessedAt());
        entity.setCreatedAt(event.getCreatedAt());
        entity.setUpdatedAt(event.getUpdatedAt());
        return entity;
    }
}
