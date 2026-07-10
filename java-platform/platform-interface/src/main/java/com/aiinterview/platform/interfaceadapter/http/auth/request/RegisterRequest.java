package com.aiinterview.platform.interfaceadapter.http.auth.request;

import jakarta.validation.constraints.NotBlank;

/**
 * 注册请求。
 */
public class RegisterRequest {

    @NotBlank(message = "username 不能为空")
    private String username;

    @NotBlank(message = "password 不能为空")
    private String password;

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}
