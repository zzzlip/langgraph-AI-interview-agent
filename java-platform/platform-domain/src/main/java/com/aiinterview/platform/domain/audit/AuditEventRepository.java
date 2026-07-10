package com.aiinterview.platform.domain.audit;

import java.util.List;

/**
 * 审计事件仓储接口。
 */
public interface AuditEventRepository {

    void save(AuditEvent auditEvent);

    /**
     * 按事件类型查询审计事件。
     */
    List<AuditEvent> findByEventType(AuditEventType eventType);

    /**
     * 按操作人 ID 查询审计事件。
     */
    List<AuditEvent> findByActorId(String actorId);
}
