package com.aiinterview.platform.application.port;

import com.aiinterview.platform.domain.csrf.CsrfToken;
import java.util.Optional;

/**
 * CSRF Token 存储接口。
 */
public interface CsrfTokenStore {

    void save(CsrfToken csrfToken);

    Optional<CsrfToken> findBySessionId(String sessionId);

    void delete(String sessionId);
}
