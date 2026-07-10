package com.aiinterview.platform.domain.auth;

/**
 * 账号角色枚举。
 */
public enum Role {

    /**
     * 普通用户。
     */
    USER("普通用户"),

    /**
     * 管理员。
     */
    ADMIN("管理员");

    private final String displayName;

    Role(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
