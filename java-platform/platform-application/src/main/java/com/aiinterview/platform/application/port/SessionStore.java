package com.aiinterview.platform.application.port;

import com.aiinterview.platform.domain.auth.Session;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * 会话存储接口。
 */
public interface SessionStore {

    String rotateSession(String currentSessionId);

    void save(Session session);

    Optional<Session> findBySessionId(String sessionId);

    void delete(String sessionId);

    List<String> findSessionIdsByAccountId(String accountId);

    void refreshExpiration(String sessionId, Instant expiresAt);
}
