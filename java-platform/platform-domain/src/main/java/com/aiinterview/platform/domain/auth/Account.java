package com.aiinterview.platform.domain.auth;

import java.time.Instant;
import java.util.Objects;

/**
 * 账号聚合根。
 */
public class Account {

    private final String id;
    private final String username;
    private final String passwordHash;
    private final Role role;
    private final AccountStatus status;
    private final Instant createdAt;
    private final Instant updatedAt;
    private final Instant deletedAt;
    private final UserLlmConfig userLlmConfig;

    /**
     * 完整构造器（含 LLM 配置）。
     */
    public Account(
            String id,
            String username,
            String passwordHash,
            Role role,
            AccountStatus status,
            Instant createdAt,
            Instant updatedAt,
            Instant deletedAt,
            UserLlmConfig userLlmConfig
    ) {
        this.id = Objects.requireNonNull(id, "id 不能为空");
        this.username = Objects.requireNonNull(username, "username 不能为空");
        this.passwordHash = Objects.requireNonNull(passwordHash, "passwordHash 不能为空");
        this.role = Objects.requireNonNull(role, "role 不能为空");
        this.status = Objects.requireNonNull(status, "status 不能为空");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt 不能为空");
        this.updatedAt = Objects.requireNonNull(updatedAt, "updatedAt 不能为空");
        this.deletedAt = deletedAt;
        this.userLlmConfig = userLlmConfig;
    }

    /**
     * 向后兼容构造器（不含 LLM 配置）。
     */
    public Account(
            String id,
            String username,
            String passwordHash,
            Role role,
            AccountStatus status,
            Instant createdAt,
            Instant updatedAt,
            Instant deletedAt
    ) {
        this(id, username, passwordHash, role, status, createdAt, updatedAt, deletedAt, null);
    }

    /**
     * 当前账号是否处于正常可用状态。
     */
    public boolean isNormal() {
        return status == AccountStatus.NORMAL && deletedAt == null;
    }

    /**
     * 当前账号是否已禁用。
     */
    public boolean isDisabled() {
        return status == AccountStatus.DISABLED;
    }

    public String getId() {
        return id;
    }

    public String getUsername() {
        return username;
    }

    public String getPasswordHash() {
        return passwordHash;
    }

    public Role getRole() {
        return role;
    }

    public AccountStatus getStatus() {
        return status;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public Instant getDeletedAt() {
        return deletedAt;
    }

    public UserLlmConfig getUserLlmConfig() {
        return userLlmConfig;
    }
}
