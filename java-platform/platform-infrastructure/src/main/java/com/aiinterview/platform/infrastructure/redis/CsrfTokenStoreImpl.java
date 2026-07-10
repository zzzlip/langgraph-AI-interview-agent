package com.aiinterview.platform.infrastructure.redis;

import com.aiinterview.platform.application.port.CsrfTokenStore;
import com.aiinterview.platform.domain.csrf.CsrfToken;
import java.time.Instant;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

/**
 * CSRF Token Redis 存储实现。
 *
 * key 格式：csrf:token:{sessionId}
 * value 格式：tokenValue|createdAtEpochMillis|expiresAtEpochMillis
 * TTL：20 分钟（与 TOKEN_TIMEOUT_SECONDS 一致）
 */
@Component
public class CsrfTokenStoreImpl implements CsrfTokenStore {

    private static final String KEY_PREFIX = "csrf:token:";
    private static final String FIELD_SEPARATOR = "|";
    private static final long TOKEN_TTL_SECONDS = 20 * 60L;

    private final StringRedisTemplate redisTemplate;

    public CsrfTokenStoreImpl(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    @Override
    public void save(CsrfToken csrfToken) {
        String key = buildKey(csrfToken.getSessionId());
        String value = csrfToken.getToken()
                + FIELD_SEPARATOR + csrfToken.getCreatedAt().toEpochMilli()
                + FIELD_SEPARATOR + csrfToken.getExpiresAt().toEpochMilli();
        redisTemplate.opsForValue().set(key, value, TOKEN_TTL_SECONDS, TimeUnit.SECONDS);
    }

    @Override
    public Optional<CsrfToken> findBySessionId(String sessionId) {
        String key = buildKey(sessionId);
        String value = redisTemplate.opsForValue().get(key);
        if (value == null || value.isEmpty()) {
            return Optional.empty();
        }

        String[] parts = value.split("\\" + FIELD_SEPARATOR, 3);
        if (parts.length < 3) {
            return Optional.empty();
        }

        try {
            String tokenValue = parts[0];
            Instant createdAt = Instant.ofEpochMilli(Long.parseLong(parts[1]));
            Instant expiresAt = Instant.ofEpochMilli(Long.parseLong(parts[2]));
            return Optional.of(new CsrfToken(tokenValue, sessionId, createdAt, expiresAt));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    @Override
    public void delete(String sessionId) {
        String key = buildKey(sessionId);
        redisTemplate.delete(key);
    }

    private static String buildKey(String sessionId) {
        return KEY_PREFIX + sessionId;
    }
}
