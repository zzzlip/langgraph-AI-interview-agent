package com.aiinterview.platform.infrastructure.redis;

import com.aiinterview.platform.application.port.SessionStore;
import com.aiinterview.platform.domain.auth.Role;
import com.aiinterview.platform.domain.auth.Session;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

/**
 * 会话 Redis 存储实现。
 */
@Component
public class SessionStoreImpl implements SessionStore {

    private static final String SESSION_KEY_PREFIX = "ai-interview:session:data:";
    private static final String ACCOUNT_INDEX_KEY_PREFIX = "ai-interview:session:account:";
    private static final String FIELD_SEPARATOR = "|";
    private static final long SESSION_TTL_SECONDS = 30 * 60L;

    private final StringRedisTemplate redisTemplate;

    public SessionStoreImpl(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    @Override
    public String rotateSession(String currentSessionId) {
        return UUID.randomUUID().toString();
    }

    @Override
    public void save(Session session) {
        String value = session.getAccountId()
                + FIELD_SEPARATOR + session.getRole().name()
                + FIELD_SEPARATOR + session.getCreatedAt().toEpochMilli()
                + FIELD_SEPARATOR + session.getExpiresAt().toEpochMilli();
        redisTemplate.opsForValue().set(sessionKey(session.getSessionId()), value, SESSION_TTL_SECONDS, TimeUnit.SECONDS);
        redisTemplate.opsForSet().add(accountIndexKey(session.getAccountId()), session.getSessionId());
        redisTemplate.expire(accountIndexKey(session.getAccountId()), SESSION_TTL_SECONDS, TimeUnit.SECONDS);
    }

    @Override
    public Optional<Session> findBySessionId(String sessionId) {
        String value = redisTemplate.opsForValue().get(sessionKey(sessionId));
        if (!StringUtils.hasText(value)) {
            return Optional.empty();
        }

        String[] parts = value.split("\\" + FIELD_SEPARATOR, 4);
        if (parts.length != 4) {
            return Optional.empty();
        }

        try {
            return Optional.of(new Session(
                    sessionId,
                    parts[0],
                    Role.valueOf(parts[1]),
                    Instant.ofEpochMilli(Long.parseLong(parts[2])),
                    Instant.ofEpochMilli(Long.parseLong(parts[3]))
            ));
        } catch (IllegalArgumentException ex) {
            return Optional.empty();
        }
    }

    @Override
    public void delete(String sessionId) {
        findBySessionId(sessionId)
                .map(Session::getAccountId)
                .ifPresent(accountId -> redisTemplate.opsForSet().remove(accountIndexKey(accountId), sessionId));
        redisTemplate.delete(sessionKey(sessionId));
    }

    @Override
    public List<String> findSessionIdsByAccountId(String accountId) {
        Set<String> sessionIds = redisTemplate.opsForSet().members(accountIndexKey(accountId));
        return sessionIds == null ? List.of() : sessionIds.stream().toList();
    }

    @Override
    public void refreshExpiration(String sessionId, Instant expiresAt) {
        findBySessionId(sessionId).ifPresent(session -> {
            Session refreshed = new Session(
                    session.getSessionId(),
                    session.getAccountId(),
                    session.getRole(),
                    session.getCreatedAt(),
                    expiresAt
            );
            save(refreshed);
        });
    }

    private static String sessionKey(String sessionId) {
        return SESSION_KEY_PREFIX + sessionId;
    }

    private static String accountIndexKey(String accountId) {
        return ACCOUNT_INDEX_KEY_PREFIX + accountId;
    }
}
