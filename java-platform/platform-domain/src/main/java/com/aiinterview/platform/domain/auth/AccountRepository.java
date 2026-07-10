package com.aiinterview.platform.domain.auth;

import java.util.Optional;

/**
 * 账号仓储接口。
 */
public interface AccountRepository {

    Optional<Account> findByUsername(String username);

    void save(Account account);

    /** 保存用户 LLM 配置。 */
    void saveLlmConfig(String accountId, UserLlmConfig llmConfig);

    /** 查询用户 LLM 配置，未配置时返回 Optional.empty()。 */
    Optional<UserLlmConfig> findLlmConfig(String accountId);
}
