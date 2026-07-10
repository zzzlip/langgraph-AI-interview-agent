package com.aiinterview.platform.infrastructure.security;

import com.aiinterview.platform.application.csrf.CsrfApplicationService;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.infrastructure.config.PlatformProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletResponse;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.web.context.SecurityContextHolderFilter;
import org.springframework.util.StringUtils;

/**
 * Security Filter Chain 配置。
 */
@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(
            HttpSecurity http,
            PlatformProperties platformProperties,
            StringRedisTemplate redisTemplate,
            ObjectMapper objectMapper,
            CsrfApplicationService csrfApplicationService,
            SessionManager sessionManager
    ) throws Exception {
        Set<String> anonymousPaths = Arrays.stream(platformProperties.getSecurity().getAnonymousPaths().split(","))
                .map(String::trim)
                .filter(StringUtils::hasText)
                .collect(Collectors.toSet());

        http.csrf(AbstractHttpConfigurer::disable)
                .cors(Customizer.withDefaults())
                .sessionManagement(session -> session
                        .sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED)
                        .sessionFixation(sessionFixation -> sessionFixation.none()))
                .exceptionHandling(exception -> exception.authenticationEntryPoint(
                        (request, response, authException) -> {
                            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                            response.setContentType("application/json");
                            response.setCharacterEncoding("UTF-8");
                            response.getWriter().write(objectMapper.writeValueAsString(
                                    BaseResult.error(20001, "未认证")
                            ));
                        }
                ))
                .authorizeHttpRequests(authorize -> {
                    anonymousPaths.forEach(path -> authorize.requestMatchers(path).permitAll());
                    authorize.requestMatchers(HttpMethod.OPTIONS, "/**").permitAll();
                    authorize.requestMatchers("/api/v1/admin/**").hasRole("ADMIN");
                    authorize.anyRequest().authenticated();
                })
                .anonymous(AbstractHttpConfigurer::disable)
                .httpBasic(AbstractHttpConfigurer::disable)
                .formLogin(AbstractHttpConfigurer::disable)
                .addFilterAfter(new RateLimitingFilter(redisTemplate, platformProperties, objectMapper), SecurityContextHolderFilter.class)
                .addFilterBefore(new SessionAuthenticationFilter(sessionManager), UsernamePasswordAuthenticationFilter.class)
                .addFilterAfter(new CsrfFilter(csrfApplicationService, objectMapper, anonymousPaths), UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }
}
