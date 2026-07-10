package com.aiinterview.platform.application.task.port;

import java.util.Map;

/**
 * 任务审计写入端口。
 */
public interface AuditWritePort {

    void writeAudit(String eventType, String actorId, String targetId, Map<String, Object> details);
}
