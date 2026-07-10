package com.aiinterview.platform.infrastructure.security;

import com.aiinterview.platform.application.csrf.CsrfApplicationService;
import com.aiinterview.platform.application.port.CsrfTokenStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 安全基础设施依赖装配。
 */
@Configuration
public class SecuritySupportConfig {

    @Bean
    public CsrfApplicationService csrfApplicationService(CsrfTokenStore csrfTokenStore) {
        return new CsrfApplicationService(csrfTokenStore);
    }
}
