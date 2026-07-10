package com.aiinterview.platform.domain.audit;

import java.time.Instant;
import java.util.Objects;

/**
 * 审计事件实体。
 */
public final class AuditEvent {

    private final String id;
    private final AuditEventType eventType;
    private final String actorId;
    private final String actorRole;
    private final String targetType;
    private final String targetId;
    private final String action;
    private final boolean success;
    private final String failureCode;
    private final String traceId;
    private final String sourceIp;
    private final String userAgent;
    private final Instant occurredAt;
    private final Instant createdAt;

    private AuditEvent(
            String id,
            AuditEventType eventType,
            String actorId,
            String actorRole,
            String targetType,
            String targetId,
            String action,
            boolean success,
            String failureCode,
            String traceId,
            String sourceIp,
            String userAgent,
            Instant occurredAt,
            Instant createdAt
    ) {
        this.id = Objects.requireNonNull(id, "id 不能为空");
        this.eventType = Objects.requireNonNull(eventType, "eventType 不能为空");
        this.actorId = Objects.requireNonNull(actorId, "actorId 不能为空");
        this.actorRole = actorRole;
        this.targetType = Objects.requireNonNull(targetType, "targetType 不能为空");
        this.targetId = Objects.requireNonNull(targetId, "targetId 不能为空");
        this.action = Objects.requireNonNull(action, "action 不能为空");
        this.traceId = Objects.requireNonNull(traceId, "traceId 不能为空");
        this.sourceIp = Objects.requireNonNull(sourceIp, "sourceIp 不能为空");
        this.success = success;
        this.failureCode = failureCode;
        this.userAgent = userAgent;
        this.occurredAt = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
    }

    // Restore 工厂用于从持久化数据反序列化
    public static AuditEvent restore(
            String id,
            AuditEventType eventType,
            String actorId,
            String actorRole,
            String targetType,
            String targetId,
            String action,
            boolean success,
            String failureCode,
            String traceId,
            String sourceIp,
            String userAgent,
            Instant occurredAt,
            Instant createdAt
    ) {
        AuditEvent event = new AuditEvent(
                id,
                eventType,
                actorId,
                actorRole,
                targetType,
                targetId,
                action,
                success,
                failureCode,
                traceId,
                sourceIp,
                userAgent,
                occurredAt,
                createdAt
        );
        return event;
    }

    /**
     * 创建成功审计事件。
     */
    public static AuditEvent success(
            String id,
            AuditEventType type,
            String actorId,
            String actorRole,
            String targetType,
            String targetId,
            String action,
            String traceId,
            String sourceIp
    ) {
        return success(id, type, actorId, actorRole, targetType, targetId, action, traceId, sourceIp, Instant.now());
    }

    /**
     * 创建成功审计事件，允许调用方传入业务时间。
     */
    public static AuditEvent success(
            String id,
            AuditEventType type,
            String actorId,
            String actorRole,
            String targetType,
            String targetId,
            String action,
            String traceId,
            String sourceIp,
            Instant occurredAt
    ) {
        Instant now = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
        return new AuditEvent(
                id,
                type,
                actorId,
                actorRole,
                targetType,
                targetId,
                action,
                true,
                null,
                traceId,
                sourceIp,
                null,
                now,
                now
        );
    }

    /**
     * 创建失败审计事件。
     */
    public static AuditEvent failure(
            String id,
            AuditEventType type,
            String actorId,
            String actorRole,
            String targetType,
            String targetId,
            String action,
            String failureCode,
            String traceId,
            String sourceIp
    ) {
        return failure(id, type, actorId, actorRole, targetType, targetId, action, failureCode, traceId, sourceIp, Instant.now());
    }

    /**
     * 创建失败审计事件，允许调用方传入业务时间。
     */
    public static AuditEvent failure(
            String id,
            AuditEventType type,
            String actorId,
            String actorRole,
            String targetType,
            String targetId,
            String action,
            String failureCode,
            String traceId,
            String sourceIp,
            Instant occurredAt
    ) {
        Instant now = Objects.requireNonNull(occurredAt, "occurredAt 不能为空");
        return new AuditEvent(
                id,
                type,
                actorId,
                actorRole,
                targetType,
                targetId,
                action,
                false,
                Objects.requireNonNull(failureCode, "failureCode 不能为空"),
                traceId,
                sourceIp,
                null,
                now,
                now
        );
    }

    public String getId() {
        return id;
    }

    public AuditEventType getEventType() {
        return eventType;
    }

    public String getActorId() {
        return actorId;
    }

    public String getActorRole() {
        return actorRole;
    }

    public String getTargetType() {
        return targetType;
    }

    public String getTargetId() {
        return targetId;
    }

    public String getAction() {
        return action;
    }

    public boolean isSuccess() {
        return success;
    }

    public String getFailureCode() {
        return failureCode;
    }

    public String getTraceId() {
        return traceId;
    }

    public String getSourceIp() {
        return sourceIp;
    }

    public String getUserAgent() {
        return userAgent;
    }

    public Instant getOccurredAt() {
        return occurredAt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
