package com.aiinterview.platform.interfaceadapter.http.auth.response;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 登录响应。
 */
public class LoginResponse {

    @JsonProperty("account_id")
    private final String accountId;

    private final String username;

    private final String role;

    @JsonProperty("csrf_token")
    private final String csrfToken;

    public LoginResponse(String accountId, String username, String role, String csrfToken) {
        this.accountId = accountId;
        this.username = username;
        this.role = role;
        this.csrfToken = csrfToken;
    }

    public String getAccountId() {
        return accountId;
    }

    public String getUsername() {
        return username;
    }

    public String getRole() {
        return role;
    }

    public String getCsrfToken() {
        return csrfToken;
    }
}
