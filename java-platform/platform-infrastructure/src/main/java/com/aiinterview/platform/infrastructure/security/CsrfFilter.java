package com.aiinterview.platform.infrastructure.security;

import com.aiinterview.platform.application.csrf.CsrfApplicationService;
import com.aiinterview.platform.domain.common.BaseResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpServletResponseWrapper;
import jakarta.servlet.http.HttpSession;
import java.io.IOException;
import java.util.Objects;
import java.util.Set;
import org.springframework.http.MediaType;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * CSRF 校验过滤器。
 */
public class CsrfFilter extends OncePerRequestFilter {

    private static final String CSRF_HEADER = "X-CSRF-Token";
    private static final int CSRF_INVALID_CODE = 70300;
    private static final int CSRF_MISSING_CODE = 70301;

    private final CsrfApplicationService csrfApplicationService;
    private final ObjectMapper objectMapper;
    private final Set<String> anonymousPaths;

    public CsrfFilter(
            CsrfApplicationService csrfApplicationService,
            ObjectMapper objectMapper,
            Set<String> anonymousPaths
    ) {
        this.csrfApplicationService = Objects.requireNonNull(csrfApplicationService, "csrfApplicationService 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
        this.anonymousPaths = Objects.requireNonNull(anonymousPaths, "anonymousPaths 不能为空");
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        if (!requiresCsrfValidation(request)) {
            filterChain.doFilter(request, response);
            return;
        }

        HttpSession session = request.getSession(false);
        String token = request.getHeader(CSRF_HEADER);
        if (token == null || token.isBlank()) {
            writeError(response, CSRF_MISSING_CODE, "CSRF Token 缺失");
            return;
        }
        if (session == null) {
            writeError(response, CSRF_INVALID_CODE, "CSRF 校验失败");
            return;
        }

        String refreshedToken;
        try {
            refreshedToken = csrfApplicationService.validateAndRefreshToken(session.getId(), token).orElse(null);
        } catch (RuntimeException ex) {
            writeError(response, CSRF_INVALID_CODE, "CSRF 校验失败");
            return;
        }
        if (refreshedToken == null) {
            writeError(response, CSRF_INVALID_CODE, "CSRF 校验失败");
            return;
        }

        CsrfHeaderResponse csrfResponse = new CsrfHeaderResponse(response, refreshedToken);
        csrfResponse.exposeToken();
        filterChain.doFilter(request, csrfResponse);
        if (!csrfResponse.isCommitted()) {
            csrfResponse.exposeToken();
        }
    }

    private boolean requiresCsrfValidation(HttpServletRequest request) {
        String method = request.getMethod();
        if ("GET".equalsIgnoreCase(method) || "HEAD".equalsIgnoreCase(method) || "OPTIONS".equalsIgnoreCase(method)) {
            return false;
        }
        return !anonymousPaths.contains(request.getRequestURI());
    }

    private void writeError(HttpServletResponse response, int code, String message) throws IOException {
        response.setStatus(HttpServletResponse.SC_FORBIDDEN);
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");
        response.getWriter().write(objectMapper.writeValueAsString(BaseResult.error(code, message)));
    }

    private static final class CsrfHeaderResponse extends HttpServletResponseWrapper {

        private final String refreshedToken;

        private CsrfHeaderResponse(HttpServletResponse response, String refreshedToken) {
            super(response);
            this.refreshedToken = refreshedToken;
        }

        private void exposeToken() {
            super.setHeader(CSRF_HEADER, refreshedToken);
        }

        @Override
        public void reset() {
            super.reset();
            exposeToken();
        }
    }
}
