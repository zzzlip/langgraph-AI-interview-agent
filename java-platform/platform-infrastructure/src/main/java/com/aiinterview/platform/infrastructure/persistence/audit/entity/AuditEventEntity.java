package com.aiinterview.platform.infrastructure.persistence.audit.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

/**
 * 审计事件数据库实体。
 */
@TableName("`audit_event`")
public class AuditEventEntity {

    @TableId
    private String id;

    @TableField("event_type")
    private String eventType;

    @TableField("actor_id")
    private String actorId;

    @TableField("actor_role")
    private String actorRole;

    @TableField("target_type")
    private String targetType;

    @TableField("target_id")
    private String targetId;

    private String action;

    private Boolean success;

    @TableField("failure_code")
    private String failureCode;

    @TableField("trace_id")
    private String traceId;

    @TableField("source_ip")
    private String sourceIp;

    @TableField("user_agent")
    private String userAgent;

    @TableField("occurred_at")
    private Instant occurredAt;

    @TableField("created_at")
    private Instant createdAt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getEventType() { return eventType; }
    public void setEventType(String eventType) { this.eventType = eventType; }

    public String getActorId() { return actorId; }
    public void setActorId(String actorId) { this.actorId = actorId; }

    public String getActorRole() { return actorRole; }
    public void setActorRole(String actorRole) { this.actorRole = actorRole; }

    public String getTargetType() { return targetType; }
    public void setTargetType(String targetType) { this.targetType = targetType; }

    public String getTargetId() { return targetId; }
    public void setTargetId(String targetId) { this.targetId = targetId; }

    public String getAction() { return action; }
    public void setAction(String action) { this.action = action; }

    public Boolean getSuccess() { return success; }
    public void setSuccess(Boolean success) { this.success = success; }

    public String getFailureCode() { return failureCode; }
    public void setFailureCode(String failureCode) { this.failureCode = failureCode; }

    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }

    public String getSourceIp() { return sourceIp; }
    public void setSourceIp(String sourceIp) { this.sourceIp = sourceIp; }

    public String getUserAgent() { return userAgent; }
    public void setUserAgent(String userAgent) { this.userAgent = userAgent; }

    public Instant getOccurredAt() { return occurredAt; }
    public void setOccurredAt(Instant occurredAt) { this.occurredAt = occurredAt; }

    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
