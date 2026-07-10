package com.aiinterview.platform.infrastructure.redis;

import org.springframework.context.annotation.Configuration;
import org.springframework.session.data.redis.config.annotation.web.http.EnableRedisHttpSession;

/**
 * Spring Session Redis 配置。
 *
 * 会话超时时间 1800 秒（30 分钟无活动超时），由 Spring Session 自动管理。
 * Redis key namespace 为 ai-interview:session，与限流和 CSRF 存储的 key 空间隔离。
 */
@Configuration
@EnableRedisHttpSession(
        maxInactiveIntervalInSeconds = 1800,
        redisNamespace = "ai-interview:session"
)
public class RedisSessionConfig {
}
