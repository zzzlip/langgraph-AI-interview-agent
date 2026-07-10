package com.aiinterview.platform.infrastructure.persistence.auth;

import com.aiinterview.platform.application.auth.AuthException;
import com.aiinterview.platform.domain.auth.Account;
import com.aiinterview.platform.domain.auth.AccountRepository;
import com.aiinterview.platform.domain.auth.AccountStatus;
import com.aiinterview.platform.domain.auth.Role;
import com.aiinterview.platform.domain.auth.UserLlmConfig;
import com.aiinterview.platform.infrastructure.persistence.auth.entity.AccountEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.stereotype.Component;

/**
 * 账号仓储基础设施实现。
 */
@Component
public class AccountRepositoryImpl implements AccountRepository {

    private static final int USERNAME_UNAVAILABLE_CODE = 70102;

    private final AccountMapper accountMapper;

    public AccountRepositoryImpl(AccountMapper accountMapper) {
        this.accountMapper = accountMapper;
    }

    @Override
    public Optional<Account> findByUsername(String username) {
        AccountEntity entity = accountMapper.selectOne(
                new LambdaQueryWrapper<AccountEntity>()
                        .eq(AccountEntity::getUsername, username)
                        .isNull(AccountEntity::getDeletedAt)
        );
        return Optional.ofNullable(entity).map(this::toDomain);
    }

    @Override
    public void save(Account account) {
        AccountEntity entity = toEntity(account);
        try {
            if (accountMapper.selectById(account.getId()) != null) {
                accountMapper.updateById(entity);
            } else {
                accountMapper.insert(entity);
            }
        } catch (DataIntegrityViolationException ex) {
            throw new AuthException(USERNAME_UNAVAILABLE_CODE, "用户名不可用，请更换后重试", ex);
        }
    }

    @Override
    public void saveLlmConfig(String accountId, UserLlmConfig llmConfig) {
        AccountEntity entity = accountMapper.selectById(accountId);
        if (entity == null) {
            throw new AuthException(20003, "账号不存在");
        }
        Map<String, Object> json = new java.util.LinkedHashMap<>();
        json.put("provider", llmConfig.getProvider());
        json.put("base_url", llmConfig.getBaseUrl());
        json.put("model", llmConfig.getModel());
        json.put("api_key", llmConfig.getApiKey());
        entity.setLlmConfig(json);
        accountMapper.updateById(entity);
    }

    @Override
    public Optional<UserLlmConfig> findLlmConfig(String accountId) {
        AccountEntity entity = accountMapper.selectById(accountId);
        if (entity == null || entity.getLlmConfig() == null) {
            return Optional.empty();
        }
        return Optional.of(toUserLlmConfig(entity.getLlmConfig()));
    }

    private UserLlmConfig toUserLlmConfig(Map<String, Object> json) {
        return new UserLlmConfig(
                stringFromJson(json, "provider"),
                stringFromJson(json, "base_url"),
                stringFromJson(json, "model"),
                stringFromJson(json, "api_key")
        );
    }

    private static String stringFromJson(Map<String, Object> json, String key) {
        Object val = json.get(key);
        return val instanceof String s ? s : "";
    }

    private Account toDomain(AccountEntity entity) {
        UserLlmConfig llmConfig = entity.getLlmConfig() != null
                ? toUserLlmConfig(entity.getLlmConfig())
                : null;
        return new Account(
                entity.getId(),
                entity.getUsername(),
                entity.getPasswordHash(),
                Role.valueOf(entity.getRole()),
                AccountStatus.valueOf(entity.getStatus()),
                entity.getCreatedAt(),
                entity.getUpdatedAt(),
                entity.getDeletedAt(),
                llmConfig
        );
    }

    private AccountEntity toEntity(Account account) {
        AccountEntity entity = new AccountEntity();
        entity.setId(account.getId());
        entity.setUsername(account.getUsername());
        entity.setPasswordHash(account.getPasswordHash());
        entity.setRole(account.getRole().name());
        entity.setStatus(account.getStatus().name());
        entity.setCreatedAt(account.getCreatedAt());
        entity.setUpdatedAt(account.getUpdatedAt());
        entity.setDeletedAt(account.getDeletedAt());
        if (account.getUserLlmConfig() != null) {
            Map<String, Object> json = new java.util.LinkedHashMap<>();
            json.put("provider", account.getUserLlmConfig().getProvider());
            json.put("base_url", account.getUserLlmConfig().getBaseUrl());
            json.put("model", account.getUserLlmConfig().getModel());
            json.put("api_key", account.getUserLlmConfig().getApiKey());
            entity.setLlmConfig(json);
        }
        return entity;
    }
}
