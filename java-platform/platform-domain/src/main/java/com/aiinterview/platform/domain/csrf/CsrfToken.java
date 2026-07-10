package com.aiinterview.platform.domain.csrf;

import java.time.Instant;
import java.util.Objects;

/**
 * CSRF Token 值对象。
 */
public final class CsrfToken {

    private final String token;
    private final String sessionId;
    private final Instant createdAt;
    private final Instant expiresAt;

    public CsrfToken(
            String token,
            String sessionId,
            Instant createdAt,
            Instant expiresAt
    ) {
        this.token = Objects.requireNonNull(token, "token 不能为空");
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.expiresAt = Objects.requireNonNull(expiresAt, "expiresAt 不能为空");
    }

    /**
     * 按当前时间判断 Token 是否已过期。
     */
    public boolean isExpired() {
        return isExpiredAt(Instant.now());
    }

    /**
     * 按指定参考时间判断 Token 是否已过期，便于测试和上层显式控制时间语义。
     */
    public boolean isExpiredAt(Instant referenceTime) {
        return Objects.requireNonNull(referenceTime, "referenceTime 不能为空").isAfter(expiresAt);
    }

    /**
     * 判断 Token 是否绑定到指定会话。
     */
    public boolean isBoundToSession(String targetSessionId) {
        return sessionId.equals(targetSessionId);
    }

    public String getToken() {
        return token;
    }

    public String getSessionId() {
        return sessionId;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getExpiresAt() {
        return expiresAt;
    }
}
