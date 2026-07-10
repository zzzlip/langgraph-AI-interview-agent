package com.aiinterview.platform.infrastructure.security;

import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.infrastructure.config.PlatformProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 限流过滤器。
 */
public class RateLimitingFilter extends OncePerRequestFilter {

    private static final int RATE_LIMIT_ERROR_CODE = 40001;
    private static final String RATE_LIMIT_ERROR_MESSAGE = "请求频率超限";
    private static final long REGISTER_RATE_LIMIT_WINDOW_SECONDS = 60 * 60L;

    private final StringRedisTemplate redisTemplate;
    private final PlatformProperties platformProperties;
    private final ObjectMapper objectMapper;

    public RateLimitingFilter(
            StringRedisTemplate redisTemplate,
            PlatformProperties platformProperties,
            ObjectMapper objectMapper
    ) {
        this.redisTemplate = Objects.requireNonNull(redisTemplate, "redisTemplate 不能为空");
        this.platformProperties = Objects.requireNonNull(platformProperties, "platformProperties 不能为空");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper 不能为空");
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        RateLimitRule rule = resolveRule(request);
        if (rule == null) {
            filterChain.doFilter(request, response);
            return;
        }

        try {
            List<String> dimensions = resolveDimensions(request, rule);
            if (isAnyDimensionLimited(request, rule, dimensions)) {
                writeRateLimitError(response);
                return;
            }
        } catch (RuntimeException ex) {
            writeRateLimitError(response);
            return;
        }

        filterChain.doFilter(request, response);
    }

    private boolean isAnyDimensionLimited(HttpServletRequest request, RateLimitRule rule, List<String> dimensions) {
        for (String dimension : dimensions) {
            long current = increaseCounter(rule, dimension);
            if (current > rule.limit()) {
                return true;
            }
        }
        return false;
    }

    private long increaseCounter(RateLimitRule rule, String dimension) {
        long window = rule.windowSeconds();
        long nowWindow = Instant.now().getEpochSecond() / window;
        String key = "ai-interview:ratelimit:" + rule.path() + ":" + dimension + ":" + nowWindow;

        Long count = redisTemplate.opsForValue().increment(key);
        if (count == null) {
            throw new IllegalStateException("限流计数失败");
        }
        if (count == 1L) {
            redisTemplate.expire(key, window, TimeUnit.SECONDS);
        }
        return count;
    }

    private String extractIp(HttpServletRequest request) {
        String xff = request.getHeader("X-Forwarded-For");
        if (xff != null && !xff.isBlank()) {
            return xff.split(",")[0].trim();
        }
        return request.getRemoteAddr() == null ? "unknown" : request.getRemoteAddr();
    }

    private String extractDevice(HttpServletRequest request) {
        String explicitDeviceId = request.getHeader("X-Device-Id");
        if (explicitDeviceId != null && !explicitDeviceId.isBlank()) {
            return "device:" + explicitDeviceId.trim();
        }
        String userAgent = request.getHeader("User-Agent");
        if (userAgent != null && !userAgent.isBlank()) {
            return "ua:" + Integer.toHexString(userAgent.hashCode());
        }
        return "device:unknown";
    }

    private List<String> resolveDimensions(HttpServletRequest request, RateLimitRule rule) {
        String ip = extractIp(request);
        List<String> dimensions = new ArrayList<>();
        dimensions.add("ip:" + ip);
        if (rule.includeDeviceDimension()) {
            dimensions.add("device:" + extractDevice(request));
        }
        return dimensions;
    }

    private RateLimitRule resolveRule(HttpServletRequest request) {
        String method = request.getMethod();
        String uri = request.getRequestURI();
        PlatformProperties.Security security = platformProperties.getSecurity();

        if ("GET".equalsIgnoreCase(method) && "/api/v1/csrf-token".equals(uri)) {
            return new RateLimitRule(uri, security.getCsrfTokenRateLimit(), security.getRateLimitWindowSeconds(), true);
        }
        if ("POST".equalsIgnoreCase(method) && "/api/v1/auth/login".equals(uri)) {
            return new RateLimitRule(uri, security.getLoginRateLimit(), security.getRateLimitWindowSeconds(), true);
        }
        if ("POST".equalsIgnoreCase(method) && "/api/v1/auth/register".equals(uri)) {
            return new RateLimitRule(uri, security.getRegisterRateLimit(), REGISTER_RATE_LIMIT_WINDOW_SECONDS, true);
        }
        return null;
    }

    private void writeRateLimitError(HttpServletResponse response) throws IOException {
        response.setStatus(HttpStatus.TOO_MANY_REQUESTS.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");
        response.getWriter().write(objectMapper.writeValueAsString(
                BaseResult.error(RATE_LIMIT_ERROR_CODE, RATE_LIMIT_ERROR_MESSAGE)
        ));
    }

    private record RateLimitRule(String path, int limit, long windowSeconds, boolean includeDeviceDimension) {
    }
}
