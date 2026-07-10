package com.aiinterview.platform.infrastructure.security;

import com.aiinterview.platform.application.session.SessionManager;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import java.io.IOException;
import java.util.List;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 基于平台会话恢复 Spring Security 认证上下文。
 */
public class SessionAuthenticationFilter extends OncePerRequestFilter {

    private static final String AUTH_ACCOUNT_ID = "AUTH_ACCOUNT_ID";
    private static final String AUTH_ROLE = "AUTH_ROLE";
    private static final String AUTH_LOGGED_OUT = "AUTH_LOGGED_OUT";

    private final SessionManager sessionManager;

    public SessionAuthenticationFilter(SessionManager sessionManager) {
        this.sessionManager = sessionManager;
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        HttpSession session = request.getSession(false);
        if (session == null) {
            filterChain.doFilter(request, response);
            return;
        }

        Boolean loggedOut = (Boolean) session.getAttribute(AUTH_LOGGED_OUT);
        String accId = stringAttribute(session, AUTH_ACCOUNT_ID);
        String role = stringAttribute(session, AUTH_ROLE);

        if (Boolean.TRUE.equals(loggedOut)) {
            filterChain.doFilter(request, response);
            return;
        }

        if (accId != null && role != null && sessionManager.getCurrentSession(session.getId()).isPresent()) {
            setAuthentication(accId, role);
        }

        try {
            filterChain.doFilter(request, response);
        } finally {
            // 整条过滤器链执行完成后再清理，避免下游过滤器读取不到认证上下文。
            SecurityContextHolder.clearContext();
        }
    }

    private static void setAuthentication(String accountId, String role) {
        UsernamePasswordAuthenticationToken authentication = new UsernamePasswordAuthenticationToken(
                accountId,
                null,
                List.of(new SimpleGrantedAuthority("ROLE_" + role))
        );
        SecurityContextHolder.getContext().setAuthentication(authentication);
    }

    private static String stringAttribute(HttpSession session, String name) {
        Object value = session.getAttribute(name);
        return value instanceof String text && !text.isBlank() ? text : null;
    }
}
