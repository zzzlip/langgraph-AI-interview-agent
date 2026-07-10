package com.aiinterview.platform.application.auth.dto;

import com.aiinterview.platform.domain.auth.Role;
import java.util.Objects;

/**
 * 登录结果 DTO。
 */
public final class LoginResult {

    private final String accountId;
    private final String username;
    private final Role role;
    private final String csrfToken;

    public LoginResult(String accountId, String username, Role role, String csrfToken) {
        this.accountId = Objects.requireNonNull(accountId, "accountId 不能为空");
        this.username = Objects.requireNonNull(username, "username 不能为空");
        this.role = Objects.requireNonNull(role, "role 不能为空");
        this.csrfToken = Objects.requireNonNull(csrfToken, "csrfToken 不能为空");
    }

    public String getAccountId() {
        return accountId;
    }

    public String getUsername() {
        return username;
    }

    public Role getRole() {
        return role;
    }

    public String getCsrfToken() {
        return csrfToken;
    }
}
