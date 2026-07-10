package com.aiinterview.platform.domain.auth;

import java.time.Instant;
import java.util.Objects;

/**
 * 会话值对象。
 */
public class Session {

    private final String sessionId;
    private final String accountId;
    private final Role role;
    private final Instant createdAt;
    private final Instant expiresAt;

    public Session(
            String sessionId,
            String accountId,
            Role role,
            Instant createdAt,
            Instant expiresAt
    ) {
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId 不能为空");
        this.accountId = Objects.requireNonNull(accountId, "accountId 不能为空");
        this.role = Objects.requireNonNull(role, "role 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.expiresAt = Objects.requireNonNull(expiresAt, "expiresAt 不能为空");
    }

    /**
     * 判断会话是否已经过期。
     */
    public boolean isExpired() {
        return isExpiredAt(Instant.now());
    }

    /**
     * 按指定参考时间判断会话是否已经过期。
     */
    public boolean isExpiredAt(Instant referenceTime) {
        return Objects.requireNonNull(referenceTime, "referenceTime 不能为空").isAfter(expiresAt);
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getAccountId() {
        return accountId;
    }

    public Role getRole() {
        return role;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getExpiresAt() {
        return expiresAt;
    }
}
