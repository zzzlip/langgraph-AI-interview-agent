package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.audit.AuditApplicationService;
import com.aiinterview.platform.application.auth.AuthApplicationService;
import com.aiinterview.platform.application.auth.UserLlmConfigApplicationService;
import com.aiinterview.platform.application.csrf.CsrfApplicationService;
import com.aiinterview.platform.application.port.PasswordEncoder;
import com.aiinterview.platform.application.port.SessionStore;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.audit.AuditEventRepository;
import com.aiinterview.platform.domain.auth.AccountRepository;
import java.security.SecureRandom;
import java.time.Clock;
import java.util.function.Supplier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 认证安全应用服务装配。
 */
@Configuration
public class SecurityApplicationConfig {

    @Bean
    public Clock clock() {
        return Clock.systemUTC();
    }

    @Bean
    public Supplier<String> idSupplier(Clock clock) {
        SecureRandom random = new SecureRandom();
        return () -> newUuidV7(clock, random);
    }

    @Bean
    public AuditApplicationService auditApplicationService(AuditEventRepository auditEventRepository) {
        return new AuditApplicationService(auditEventRepository);
    }

    @Bean
    public SessionManager sessionManager(
            SessionStore sessionStore,
            Clock clock,
            CsrfApplicationService csrfApplicationService
    ) {
        return new SessionManager(sessionStore, clock, csrfApplicationService);
    }

    @Bean
    public UserLlmConfigApplicationService userLlmConfigApplicationService(
            AccountRepository accountRepository
    ) {
        return new UserLlmConfigApplicationService(accountRepository);
    }

    @Bean
    public AuthApplicationService authApplicationService(
            AccountRepository accountRepository,
            PasswordEncoder passwordEncoder,
            SessionManager sessionManager,
            CsrfApplicationService csrfApplicationService,
            AuditApplicationService auditApplicationService,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        return new AuthApplicationService(
                accountRepository,
                passwordEncoder,
                sessionManager,
                csrfApplicationService,
                auditApplicationService,
                clock,
                idSupplier
        );
    }

    private static String newUuidV7(Clock clock, SecureRandom random) {
        long timestampMillis = clock.millis() & 0x0000_FFFF_FFFF_FFFFL;
        long randomA = random.nextInt(1 << 12);
        long mostSignificantBits = (timestampMillis << 16) | 0x7000L | randomA;
        long leastSignificantBits = 0x8000_0000_0000_0000L | (random.nextLong() & 0x3FFF_FFFF_FFFF_FFFFL);
        return new java.util.UUID(mostSignificantBits, leastSignificantBits).toString();
    }
}
