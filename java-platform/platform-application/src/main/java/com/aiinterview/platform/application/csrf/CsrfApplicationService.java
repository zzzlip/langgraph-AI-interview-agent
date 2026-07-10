package com.aiinterview.platform.application.csrf;

import com.aiinterview.platform.application.common.BaseApplicationService;
import com.aiinterview.platform.application.port.CsrfTokenStore;
import com.aiinterview.platform.domain.csrf.CsrfToken;
import java.time.Clock;
import java.time.Instant;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * CSRF Token 应用服务。
 */
public class CsrfApplicationService extends BaseApplicationService {

    static final long TOKEN_TIMEOUT_SECONDS = 20 * 60L;

    private final CsrfTokenStore csrfTokenStore;
    private final Clock clock;
    private final Supplier<String> tokenSupplier;

    public CsrfApplicationService(CsrfTokenStore csrfTokenStore) {
        this(csrfTokenStore, Clock.systemUTC(), () -> java.util.UUID.randomUUID().toString());
    }

    public CsrfApplicationService(CsrfTokenStore csrfTokenStore, Clock clock, Supplier<String> tokenSupplier) {
        this.csrfTokenStore = Objects.requireNonNull(csrfTokenStore, "csrfTokenStore 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.tokenSupplier = Objects.requireNonNull(tokenSupplier, "tokenSupplier 不能为空");
    }

    /**
     * 为指定会话生成 CSRF Token。
     */
    public CsrfToken generateToken(String sessionId) {
        return generateToken(sessionId, null);
    }

    private CsrfToken generateToken(String sessionId, String previousTokenValue) {
        Instant now = Instant.now(clock);
        String tokenValue = nextTokenValue(previousTokenValue);
        CsrfToken csrfToken = new CsrfToken(
                tokenValue,
                sessionId,
                now,
                now.plusSeconds(TOKEN_TIMEOUT_SECONDS)
        );
        csrfTokenStore.save(csrfToken);
        return csrfToken;
    }

    /**
     * 刷新指定会话的 CSRF Token。
     */
    public CsrfToken refreshToken(String sessionId) {
        Optional<CsrfToken> existingToken = csrfTokenStore.findBySessionId(sessionId);
        csrfTokenStore.delete(sessionId);
        return generateToken(sessionId, existingToken.map(CsrfToken::getToken).orElse(null));
    }

    /**
     * 校验指定会话的 CSRF Token。
     */
    public boolean validateToken(String sessionId, String tokenValue) {
        return validateAndRefreshToken(sessionId, tokenValue).isPresent();
    }

    /**
     * 校验并轮换指定会话的 CSRF Token，返回轮换后的新 Token。
     */
    public Optional<String> validateAndRefreshToken(String sessionId, String tokenValue) {
        Optional<CsrfToken> storedToken = csrfTokenStore.findBySessionId(sessionId);
        if (storedToken.isEmpty()) {
            return Optional.empty();
        }

        CsrfToken csrfToken = storedToken.orElseThrow();
        Instant now = Instant.now(clock);

        if (csrfToken.isExpiredAt(now)) {
            return Optional.empty();
        }

        if (!csrfToken.isBoundToSession(sessionId) || !Objects.equals(csrfToken.getToken(), tokenValue)) {
            return Optional.empty();
        }

        CsrfToken refreshed = refreshToken(sessionId);
        return Optional.of(refreshed.getToken());
    }

    private String nextTokenValue(String previousTokenValue) {
        String tokenValue = tokenSupplier.get();

        if (previousTokenValue == null) {
            return tokenValue;
        }

        while (Objects.equals(previousTokenValue, tokenValue)) {
            tokenValue = tokenSupplier.get();
        }

        return tokenValue;
    }

    /**
     * 使指定会话的 CSRF Token 失效。
     */
    public void invalidateToken(String sessionId) {
        csrfTokenStore.delete(sessionId);
    }
}
