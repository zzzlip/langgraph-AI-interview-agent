package com.aiinterview.platform.application.auth.command;

import java.util.Objects;

/**
 * 注册命令。
 */
public final class RegisterCommand {

    private final String username;
    private final String password;

    public RegisterCommand(String username, String password) {
        this.username = Objects.requireNonNull(username, "username 不能为空");
        this.password = Objects.requireNonNull(password, "password 不能为空");
    }

    public String getUsername() {
        return username;
    }

    public String getPassword() {
        return password;
    }
}
