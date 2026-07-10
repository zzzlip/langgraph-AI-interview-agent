package com.aiinterview.platform.domain.auth;

/**
 * 账号状态枚举。
 */
public enum AccountStatus {

    /**
     * 正常状态。
     */
    NORMAL("正常"),

    /**
     * 已禁用状态。
     */
    DISABLED("已禁用");

    private final String displayName;

    AccountStatus(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
