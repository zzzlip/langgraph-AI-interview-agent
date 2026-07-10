package com.aiinterview.platform.interfaceadapter.http.support;

import com.aiinterview.platform.application.security.SecurityException;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.auth.Role;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import java.util.Optional;

/**
 * HTTP 会话、角色和 CSRF 的统一校验入口。
 */
public final class HttpSessionSupport {

    private static final int SESSION_INVALID_CODE = 70201;
    private static final int PERMISSION_DENIED_CODE = 20003;
    private static final int CSRF_MISSING_CODE = 70301;

    private HttpSessionSupport() {
    }

    public static CurrentSession requireUserSession(HttpServletRequest request, SessionManager sessionManager) {
        return requireRole(request.getSession(false), sessionManager, Role.USER);
    }

    public static CurrentSession requireUserSession(HttpSession session, SessionManager sessionManager) {
        return requireRole(session, sessionManager, Role.USER);
    }

    public static CurrentSession requireAdminSession(HttpServletRequest request, SessionManager sessionManager) {
        return requireRole(request.getSession(false), sessionManager, Role.ADMIN);
    }

    public static void requireCsrfToken(String csrfToken) {
        if (csrfToken == null || csrfToken.isBlank()) {
            throw new SecurityException(CSRF_MISSING_CODE, "CSRF Token 缺失");
        }
    }

    private static CurrentSession requireRole(HttpSession session, SessionManager sessionManager, Role role) {
        if (session == null) {
            throw new SecurityException(SESSION_INVALID_CODE, "会话已过期");
        }
        Optional<CurrentSession> currentSession = sessionManager.getCurrentSession(session.getId());
        if (currentSession.isEmpty()) {
            throw new SecurityException(SESSION_INVALID_CODE, "会话已过期");
        }
        if (currentSession.get().getRole() != role) {
            throw new SecurityException(PERMISSION_DENIED_CODE, "无权限");
        }
        return currentSession.get();
    }
}
