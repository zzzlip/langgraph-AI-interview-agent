package com.aiinterview.platform.application.session;

import com.aiinterview.platform.application.common.BaseApplicationService;
import com.aiinterview.platform.application.csrf.CsrfApplicationService;
import com.aiinterview.platform.application.port.SessionStore;
import com.aiinterview.platform.domain.auth.Role;
import com.aiinterview.platform.domain.auth.Session;
import java.time.Clock;
import java.time.Instant;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * 会话管理应用服务。
 */
public class SessionManager extends BaseApplicationService {

    static final long SESSION_TIMEOUT_SECONDS = 30 * 60L;

    private final SessionStore sessionStore;
    private final Clock clock;
    private final CsrfApplicationService csrfApplicationService;

    public SessionManager(SessionStore sessionStore) {
        this(sessionStore, Clock.systemUTC(), null);
    }

    public SessionManager(SessionStore sessionStore, Clock clock) {
        this(sessionStore, clock, null);
    }

    public SessionManager(SessionStore sessionStore, Clock clock, CsrfApplicationService csrfApplicationService) {
        this.sessionStore = Objects.requireNonNull(sessionStore, "sessionStore 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.csrfApplicationService = csrfApplicationService;
    }

    /**
     * 创建新会话并轮换会话标识。
     */
    public Session createSession(String accountId, Role role, String currentSessionId) {
        Instant now = Instant.now(clock);
        String sessionId = (currentSessionId == null || currentSessionId.isBlank())
                ? sessionStore.rotateSession(currentSessionId)
                : currentSessionId;
        Session session = new Session(
                sessionId,
                accountId,
                role,
                now,
                now.plusSeconds(SESSION_TIMEOUT_SECONDS)
        );

        sessionStore.save(session);
        return session;
    }

    /**
     * 使指定会话失效。
     */
    public void invalidateSession(String sessionId) {
        sessionStore.delete(sessionId);
        if (csrfApplicationService != null) {
            csrfApplicationService.invalidateToken(sessionId);
        }
    }

    /**
     * 使账号关联的全部会话失效。
     */
    public void invalidateSessionsByAccountId(String accountId) {
        List<String> sessionIds = sessionStore.findSessionIdsByAccountId(accountId);
        for (String sessionId : sessionIds) {
            sessionStore.delete(sessionId);
            if (csrfApplicationService != null) {
                csrfApplicationService.invalidateToken(sessionId);
            }
        }
    }

    /**
     * 获取当前会话快照。
     */
    public Optional<CurrentSession> getCurrentSession(String sessionId) {
        Instant now = Instant.now(clock);

        return sessionStore.findBySessionId(sessionId)
                .filter(session -> !session.isExpiredAt(now))
                .map(session -> new CurrentSession(
                        session.getSessionId(),
                        session.getAccountId(),
                        session.getRole(),
                        session.getCreatedAt()
                ));
    }

    /**
     * 刷新会话过期时间。
     */
    public void refreshExpiration(String sessionId) {
        sessionStore.refreshExpiration(
                sessionId,
                Instant.now(clock).plusSeconds(SESSION_TIMEOUT_SECONDS)
        );
    }
}
