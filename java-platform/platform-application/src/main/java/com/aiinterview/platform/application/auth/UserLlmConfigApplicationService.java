package com.aiinterview.platform.application.auth;

import com.aiinterview.platform.application.auth.command.SaveLlmConfigCommand;
import com.aiinterview.platform.application.auth.dto.UserLlmConfigResult;
import com.aiinterview.platform.application.llm.LlmConfig;
import com.aiinterview.platform.domain.auth.AccountRepository;
import com.aiinterview.platform.domain.auth.UserLlmConfig;
import java.util.Optional;

/**
 * 用户 LLM 配置应用服务。
 */
public class UserLlmConfigApplicationService {

    private static final int LLM_CONFIG_INVALID_CODE = 70103;
    private static final String SUPPORTED_PROVIDER = "deepseek";

    private final AccountRepository accountRepository;

    public UserLlmConfigApplicationService(AccountRepository accountRepository) {
        this.accountRepository = accountRepository;
    }

    /**
     * 保存用户 LLM 配置。
     */
    public UserLlmConfigResult saveLlmConfig(String accountId, SaveLlmConfigCommand command) {
        String provider = normalizeProvider(command.getProvider());
        String baseUrl = normalizeBaseUrl(command.getBaseUrl());
        String model = normalizeModel(command.getModel());
        Optional<UserLlmConfig> existing = accountRepository.findLlmConfig(accountId);
        String apiKey = normalizeOptional(command.getApiKey());
        if (apiKey == null) {
            apiKey = existing.map(UserLlmConfig::getApiKey)
                    .orElseThrow(() -> invalid("首次保存 DeepSeek 配置必须填写 API Key"));
        }
        UserLlmConfig config = new UserLlmConfig(
                provider,
                baseUrl,
                model,
                apiKey
        );
        accountRepository.saveLlmConfig(accountId, config);
        return toResult(config);
    }

    /**
     * 查询用户 LLM 配置，返回不含 api_key 的公开结果。
     */
    public Optional<UserLlmConfigResult> getLlmConfig(String accountId) {
        return accountRepository.findLlmConfig(accountId)
                .map(this::toResult);
    }

    /**
     * 解析任务级 LLM 配置。
     * 缺少 api_key 时只能复用与已保存公开配置完全一致的用户密钥。
     */
    public LlmConfig resolveTaskLlmConfig(
            String accountId,
            String providerInput,
            String baseUrlInput,
            String modelInput,
            String apiKeyInput
    ) {
        String provider = normalizeProvider(providerInput);
        String baseUrl = normalizeBaseUrl(baseUrlInput);
        String model = normalizeModel(modelInput);
        String apiKey = normalizeOptional(apiKeyInput);
        if (apiKey != null) {
            return new LlmConfig(provider, baseUrl, model, apiKey);
        }
        // 任务请求不带 api_key 时，只能复用同一用户已保存且公开参数完全一致的私密 key。
        UserLlmConfig saved = accountRepository.findLlmConfig(accountId)
                .orElseThrow(() -> invalid("未保存 DeepSeek API Key，请先完成 LLM 配置"));
        if (!saved.getProvider().equals(provider)
                || !saved.getBaseUrl().equals(baseUrl)
                || !saved.getModel().equals(model)) {
            throw invalid("llm_config 与已保存配置不一致，请先保存配置");
        }
        return new LlmConfig(provider, baseUrl, model, saved.getApiKey());
    }

    private UserLlmConfigResult toResult(UserLlmConfig config) {
        return new UserLlmConfigResult(
                config.getProvider(),
                config.getBaseUrl(),
                config.getModel(),
                true
        );
    }

    private static String normalizeProvider(String provider) {
        String normalized = normalizeOptional(provider);
        if (normalized == null || !SUPPORTED_PROVIDER.equals(normalized.toLowerCase())) {
            throw invalid("当前仅支持 deepseek");
        }
        return SUPPORTED_PROVIDER;
    }

    private static String normalizeBaseUrl(String baseUrl) {
        String normalized = normalizeOptional(baseUrl);
        if (normalized == null || !normalized.startsWith("https://")) {
            throw invalid("base_url 必须是 https 地址");
        }
        return normalized;
    }

    private static String normalizeModel(String model) {
        String normalized = normalizeOptional(model);
        if (normalized == null) {
            throw invalid("model 不能为空");
        }
        return normalized;
    }

    private static String normalizeOptional(String value) {
        if (value == null) {
            return null;
        }
        String normalized = value.trim();
        return normalized.isEmpty() ? null : normalized;
    }

    private static AuthException invalid(String message) {
        return new AuthException(LLM_CONFIG_INVALID_CODE, message);
    }
}
