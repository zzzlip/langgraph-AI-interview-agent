package com.aiinterview.platform.application.auth;

import com.aiinterview.platform.application.audit.AuditApplicationService;
import com.aiinterview.platform.application.audit.AuditReliability;
import com.aiinterview.platform.application.auth.command.LoginCommand;
import com.aiinterview.platform.application.auth.command.RegisterCommand;
import com.aiinterview.platform.application.auth.dto.LoginResult;
import com.aiinterview.platform.application.auth.dto.RegisterResult;
import com.aiinterview.platform.application.common.BaseApplicationService;
import com.aiinterview.platform.application.csrf.CsrfApplicationService;
import com.aiinterview.platform.application.port.PasswordEncoder;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.audit.AuditEvent;
import com.aiinterview.platform.domain.audit.AuditEventType;
import com.aiinterview.platform.domain.auth.Account;
import com.aiinterview.platform.domain.auth.AccountRepository;
import com.aiinterview.platform.domain.auth.AccountStatus;
import com.aiinterview.platform.domain.auth.PasswordPolicy;
import com.aiinterview.platform.domain.auth.Role;
import java.time.Clock;
import java.time.Instant;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.regex.Pattern;

/**
 * 认证应用服务。
 * 负责注册、登录、登出用例编排。
 */
public class AuthApplicationService extends BaseApplicationService {

    private static final Pattern USERNAME_PATTERN = Pattern.compile("^[a-zA-Z0-9_-]{3,64}$");

    private static final int USERNAME_FORMAT_INVALID_CODE = 70100;
    private static final int PASSWORD_WEAK_CODE = 70101;
    private static final int USERNAME_UNAVAILABLE_CODE = 70102;
    private static final int LOGIN_FAILED_CODE = 70200;

    private final AccountRepository accountRepository;
    private final PasswordEncoder passwordEncoder;
    private final SessionManager sessionManager;
    private final CsrfApplicationService csrfApplicationService;
    private final AuditApplicationService auditApplicationService;
    private final Clock clock;
    private final Supplier<String> idSupplier;

    public AuthApplicationService(
            AccountRepository accountRepository,
            PasswordEncoder passwordEncoder,
            SessionManager sessionManager,
            CsrfApplicationService csrfApplicationService,
            AuditApplicationService auditApplicationService,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this.accountRepository = Objects.requireNonNull(accountRepository, "accountRepository 不能为空");
        this.passwordEncoder = Objects.requireNonNull(passwordEncoder, "passwordEncoder 不能为空");
        this.sessionManager = Objects.requireNonNull(sessionManager, "sessionManager 不能为空");
        this.csrfApplicationService = Objects.requireNonNull(csrfApplicationService, "csrfApplicationService 不能为空");
        this.auditApplicationService = Objects.requireNonNull(auditApplicationService, "auditApplicationService 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
    }

    /**
     * 用户注册。
     */
    public RegisterResult register(RegisterCommand cmd) {
        String username = cmd.getUsername().trim();

        if (!isUsernameValid(username)) {
            auditApplicationService.recordEvent(
                    AuditEvent.failure(idSupplier.get(), AuditEventType.REGISTER_FAILURE, "anonymous", null,
                            "account", username, "register", String.valueOf(USERNAME_FORMAT_INVALID_CODE),
                            currentTraceId(), currentSourceIp(), Instant.now(clock)),
                    AuditReliability.MANDATORY
            );
            throw new AuthException(USERNAME_FORMAT_INVALID_CODE, "用户名格式无效，长度 3-64 位，只允许字母、数字、下划线、连字符");
        }

        if (!PasswordPolicy.isValid(cmd.getPassword())) {
            auditApplicationService.recordEvent(
                    AuditEvent.failure(idSupplier.get(), AuditEventType.REGISTER_FAILURE, "anonymous", null,
                            "account", username, "register", String.valueOf(PASSWORD_WEAK_CODE),
                            currentTraceId(), currentSourceIp(), Instant.now(clock)),
                    AuditReliability.MANDATORY
            );
            throw new AuthException(PASSWORD_WEAK_CODE, "密码不满足强度要求，至少包含大写字母、小写字母、数字各一个，长度 8-128 位");
        }

        if (accountRepository.findByUsername(normalizeUsername(username)).isPresent()) {
            auditApplicationService.recordEvent(
                    AuditEvent.failure(idSupplier.get(), AuditEventType.REGISTER_FAILURE, "anonymous", null,
                            "account", username, "register", String.valueOf(USERNAME_UNAVAILABLE_CODE),
                            currentTraceId(), currentSourceIp(), Instant.now(clock)),
                    AuditReliability.MANDATORY
            );
            throw new AuthException(USERNAME_UNAVAILABLE_CODE, "用户名不可用，请更换后重试");
        }

        Instant now = Instant.now(clock);
        Account account = new Account(
                idSupplier.get(),
                username,
                passwordEncoder.encode(cmd.getPassword()),
                Role.USER,
                AccountStatus.NORMAL,
                now,
                now,
                null,
                null
        );

        accountRepository.save(account);

        auditApplicationService.recordEvent(
                AuditEvent.success(idSupplier.get(), AuditEventType.REGISTER_SUCCESS, account.getId(), Role.USER.name(),
                        "account", account.getId(), "register", currentTraceId(), currentSourceIp(), now),
                AuditReliability.MANDATORY
        );

        return RegisterResult.SUCCESS;
    }

    /**
     * 用户登录。
     */
    public LoginResult login(LoginCommand cmd, String currentSessionId, String previousSessionId) {
        String username = cmd.getUsername().trim();

        Account account = accountRepository.findByUsername(normalizeUsername(username)).orElse(null);

        if (account == null || !passwordEncoder.matches(cmd.getPassword(), account.getPasswordHash())) {
            recordLoginFailure("anonymous", normalizeUsername(username));
            throw new AuthException(LOGIN_FAILED_CODE, "用户名或密码错误");
        }

        if (!account.isNormal()) {
            recordLoginFailure("anonymous", normalizeUsername(username));
            throw new AuthException(LOGIN_FAILED_CODE, "用户名或密码错误");
        }

        // 清理轮换前会话的 CSRF Token，避免旧会话 Token 在并发场景下被复用
        if (previousSessionId != null
                && !previousSessionId.isEmpty()
                && !previousSessionId.equals(currentSessionId)) {
            csrfApplicationService.invalidateToken(previousSessionId);
        }

        com.aiinterview.platform.domain.auth.Session session = sessionManager.createSession(
                account.getId(), account.getRole(), currentSessionId
        );

        com.aiinterview.platform.domain.csrf.CsrfToken csrfToken = csrfApplicationService.refreshToken(session.getSessionId());

        auditApplicationService.recordEvent(
                AuditEvent.success(idSupplier.get(), AuditEventType.LOGIN_SUCCESS, account.getId(), account.getRole().name(),
                        "account", account.getId(), "login", currentTraceId(), currentSourceIp(), Instant.now(clock)),
                AuditReliability.MANDATORY
        );

        return new LoginResult(account.getId(), account.getUsername(), account.getRole(), csrfToken.getToken());
    }

    /**
     * 用户登出。
     */
    public void logout(String sessionId, String actorId, String actorRole) {
        sessionManager.invalidateSession(sessionId);

        auditApplicationService.recordEvent(
                AuditEvent.success(idSupplier.get(), AuditEventType.LOGOUT, actorId, actorRole,
                        "session", sessionId, "logout", currentTraceId(), currentSourceIp(), Instant.now(clock)),
                AuditReliability.MANDATORY
        );
    }

    private void recordLoginFailure(String actorId, String username) {
        auditApplicationService.recordEvent(
                AuditEvent.failure(idSupplier.get(), AuditEventType.LOGIN_FAILURE, actorId, null,
                        "account", username, "login", String.valueOf(LOGIN_FAILED_CODE),
                        currentTraceId(), currentSourceIp(), Instant.now(clock)),
                AuditReliability.MANDATORY
        );
    }

    private static boolean isUsernameValid(String username) {
        return username != null && USERNAME_PATTERN.matcher(username).matches();
    }

    private static String normalizeUsername(String username) {
        return username.toLowerCase();
    }
}
