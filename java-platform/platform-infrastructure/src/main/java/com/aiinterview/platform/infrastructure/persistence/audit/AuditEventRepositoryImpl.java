package com.aiinterview.platform.infrastructure.persistence.audit;

import com.aiinterview.platform.domain.audit.AuditEvent;
import com.aiinterview.platform.domain.audit.AuditEventRepository;
import com.aiinterview.platform.domain.audit.AuditEventType;
import com.aiinterview.platform.infrastructure.persistence.audit.entity.AuditEventEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.stereotype.Component;

/**
 * 审计事件仓储基础设施实现。
 */
@Component
public class AuditEventRepositoryImpl implements AuditEventRepository {

    private final AuditEventMapper auditEventMapper;

    public AuditEventRepositoryImpl(AuditEventMapper auditEventMapper) {
        this.auditEventMapper = auditEventMapper;
    }

    @Override
    public void save(AuditEvent auditEvent) {
        AuditEventEntity entity = toEntity(auditEvent);
        auditEventMapper.insert(entity);
    }

    @Override
    public List<AuditEvent> findByEventType(AuditEventType eventType) {
        List<AuditEventEntity> entities = auditEventMapper.selectList(
                new LambdaQueryWrapper<AuditEventEntity>()
                        .eq(AuditEventEntity::getEventType, eventType.name())
        );
        return entities.stream().map(this::toDomain).collect(Collectors.toList());
    }

    @Override
    public List<AuditEvent> findByActorId(String actorId) {
        List<AuditEventEntity> entities = auditEventMapper.selectList(
                new LambdaQueryWrapper<AuditEventEntity>()
                        .eq(AuditEventEntity::getActorId, actorId)
        );
        return entities.stream().map(this::toDomain).collect(Collectors.toList());
    }

    private AuditEvent toDomain(AuditEventEntity entity) {
        return AuditEvent.restore(
                entity.getId(),
                AuditEventType.valueOf(entity.getEventType()),
                entity.getActorId(),
                entity.getActorRole(),
                entity.getTargetType(),
                entity.getTargetId(),
                entity.getAction(),
                entity.getSuccess(),
                entity.getFailureCode(),
                entity.getTraceId(),
                entity.getSourceIp(),
                entity.getUserAgent(),
                entity.getOccurredAt(),
                entity.getCreatedAt()
        );
    }

    private AuditEventEntity toEntity(AuditEvent event) {
        AuditEventEntity entity = new AuditEventEntity();
        entity.setId(event.getId());
        entity.setEventType(event.getEventType().name());
        entity.setActorId(event.getActorId());
        entity.setActorRole(event.getActorRole());
        entity.setTargetType(event.getTargetType());
        entity.setTargetId(event.getTargetId());
        entity.setAction(event.getAction());
        entity.setSuccess(event.isSuccess());
        entity.setFailureCode(event.getFailureCode());
        entity.setTraceId(event.getTraceId());
        entity.setSourceIp(event.getSourceIp());
        entity.setUserAgent(event.getUserAgent());
        entity.setOccurredAt(event.getOccurredAt());
        entity.setCreatedAt(event.getCreatedAt());
        return entity;
    }
}
