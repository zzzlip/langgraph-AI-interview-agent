package com.aiinterview.platform.application.audit;

import com.aiinterview.platform.application.common.BaseApplicationService;
import com.aiinterview.platform.domain.audit.AuditEvent;
import com.aiinterview.platform.domain.audit.AuditEventRepository;
import java.util.Objects;

/**
 * 审计事件应用服务。
 */
public class AuditApplicationService extends BaseApplicationService {

    private final AuditEventRepository auditEventRepository;

    public AuditApplicationService(AuditEventRepository auditEventRepository) {
        this.auditEventRepository = Objects.requireNonNull(auditEventRepository, "auditEventRepository 不能为空");
    }

    /**
     * 按可靠性策略写入审计事件。
     */
    public void recordEvent(AuditEvent event, AuditReliability reliability) {
        Objects.requireNonNull(event, "event 不能为空");
        Objects.requireNonNull(reliability, "reliability 不能为空");

        try {
            auditEventRepository.save(event);
        } catch (RuntimeException ex) {
            if (reliability == AuditReliability.MANDATORY) {
                throw new AuditWriteException("审计事件写入失败", ex);
            }

            log.warn("审计事件写入失败，按 BEST_EFFORT 忽略: eventType={}, traceId={}",
                    event.getEventType(), event.getTraceId(), ex);
        }
    }
}
