package com.aiinterview.platform.application.security;

import com.aiinterview.platform.domain.auth.Role;
import java.util.Objects;

/**
 * 安全策略。
 */
public class SecurityPolicy {

    private static final int FORBIDDEN_CODE = 20003;
    private static final String FORBIDDEN_MESSAGE = "无权限";

    /**
     * 判断资源是否归当前用户所有。
     */
    public boolean isResourceOwner(String resourceUserId, String currentUserId) {
        return Objects.equals(resourceUserId, currentUserId);
    }

    /**
     * 判断当前角色是否为管理员。
     */
    public boolean isAdmin(Role role) {
        return Role.ADMIN == role;
    }

    /**
     * 资源归属校验，不通过时抛出无权限异常。
     */
    public void requireResourceOwner(String resourceUserId, String currentUserId) {
        if (!isResourceOwner(resourceUserId, currentUserId)) {
            throw new SecurityException(FORBIDDEN_CODE, FORBIDDEN_MESSAGE);
        }
    }

    /**
     * 管理员角色校验，不通过时抛出无权限异常。
     */
    public void requireAdmin(Role role) {
        if (!isAdmin(role)) {
            throw new SecurityException(FORBIDDEN_CODE, FORBIDDEN_MESSAGE);
        }
    }

    /**
     * 资源存在性掩码校验。
     */
    public void requireResourceAccess(boolean resourceExists, boolean isOwner) {
        if (!resourceExists || !isOwner) {
            throw new ResourceNotFoundException();
        }
    }
}
