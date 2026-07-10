package com.aiinterview.platform.application.session;

import com.aiinterview.platform.domain.auth.Role;
import java.time.Instant;
import java.util.Objects;

/**
 * 当前会话上下文。
 */
public final class CurrentSession {

    private final String sessionId;
    private final String accountId;
    private final Role role;
    private final Instant createdAt;

    public CurrentSession(String sessionId, String accountId, Role role, Instant createdAt) {
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId 不能为空");
        this.accountId = Objects.requireNonNull(accountId, "accountId 不能为空");
        this.role = Objects.requireNonNull(role, "role 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
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
}
