package com.aiinterview.platform.interfaceadapter.http.auth;

import com.aiinterview.platform.application.auth.AuthApplicationService;
import com.aiinterview.platform.application.auth.command.LoginCommand;
import com.aiinterview.platform.application.auth.command.RegisterCommand;
import com.aiinterview.platform.application.auth.dto.LoginResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.http.auth.request.LoginRequest;
import com.aiinterview.platform.interfaceadapter.http.auth.request.RegisterRequest;
import com.aiinterview.platform.interfaceadapter.http.auth.response.LoginResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import java.util.Optional;
import org.springframework.http.CacheControl;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

/**
 * 认证接口。
 */
@RestController
@RequestMapping("/api/v1/auth")
public class AuthController {

    private static final String AUTH_SESSION_MARKER = "AUTHENTICATED";
    private static final String AUTH_ACCOUNT_ID = "AUTH_ACCOUNT_ID";
    private static final String AUTH_ROLE = "AUTH_ROLE";
    private static final String AUTH_LOGGED_OUT = "AUTH_LOGGED_OUT";
    private static final int SESSION_INVALID_CODE = 70201;

    private final AuthApplicationService authApplicationService;
    private final SessionManager sessionManager;

    public AuthController(
            AuthApplicationService authApplicationService,
            SessionManager sessionManager
    ) {
        this.authApplicationService = authApplicationService;
        this.sessionManager = sessionManager;
    }

    @PostMapping("/register")
    @ResponseStatus(HttpStatus.CREATED)
    public ResponseEntity<BaseResult<Void>> register(@Valid @RequestBody RegisterRequest request) {
        authApplicationService.register(new RegisterCommand(request.getUsername(), request.getPassword()));
        return ResponseEntity.status(HttpStatus.CREATED)
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success());
    }

    @PostMapping("/login")
    public ResponseEntity<BaseResult<LoginResponse>> login(
            @Valid @RequestBody LoginRequest request,
            HttpServletRequest servletRequest
    ) {
        HttpSession session = servletRequest.getSession(true);
        String previousSessionId = session.getId();
        // Servlet 容器先完成会话 ID 轮换，应用层使用轮换后的 ID 落库。
        String rotatedSessionId = servletRequest.changeSessionId();

        if (session != null) {
            session.setAttribute(AUTH_SESSION_MARKER, Boolean.TRUE);
        }
        LoginResult loginResult = authApplicationService.login(
                new LoginCommand(request.getUsername(), request.getPassword()),
                rotatedSessionId,
                previousSessionId
        );
        if (session != null) {
            session.setAttribute(AUTH_ACCOUNT_ID, loginResult.getAccountId());
            session.setAttribute(AUTH_ROLE, loginResult.getRole().name());
        }
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(new LoginResponse(
                        loginResult.getAccountId(),
                        loginResult.getUsername(),
                        loginResult.getRole().name(),
                        loginResult.getCsrfToken()
                )));
    }

    @PostMapping("/logout")
    public ResponseEntity<BaseResult<Void>> logout(
            HttpSession session
    ) {
        String sessionId = session == null ? null : session.getId();
        if (sessionId == null || sessionId.isBlank()) {
            throw new com.aiinterview.platform.application.security.SecurityException(SESSION_INVALID_CODE, "会话已过期");
        }

        Optional<CurrentSession> currentSession = sessionManager.getCurrentSession(sessionId);
        if (currentSession.isEmpty()) {
            throw new com.aiinterview.platform.application.security.SecurityException(SESSION_INVALID_CODE, "会话已过期");
        }

        authApplicationService.logout(
                sessionId,
                currentSession.get().getAccountId(),
                currentSession.get().getRole().name()
        );
        session.setAttribute(AUTH_LOGGED_OUT, Boolean.TRUE);
        session.removeAttribute(AUTH_SESSION_MARKER);
        session.removeAttribute(AUTH_ACCOUNT_ID);
        session.removeAttribute(AUTH_ROLE);
        session.invalidate();
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success());
    }
}
