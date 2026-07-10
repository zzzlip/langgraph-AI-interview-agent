package com.aiinterview.platform.domain.auth;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 用户持久化的 LLM 配置值对象，存储在 account.llm_config JSON 列中。
 * 仅支持 DeepSeek。
 */
public class UserLlmConfig {

    private final String provider;
    private final String baseUrl;
    private final String model;
    private final String apiKey;

    public UserLlmConfig(String provider, String baseUrl, String model, String apiKey) {
        this.provider = Objects.requireNonNull(provider, "provider 不能为空");
        this.baseUrl = Objects.requireNonNull(baseUrl, "baseUrl 不能为空");
        this.model = Objects.requireNonNull(model, "model 不能为空");
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey 不能为空");
    }

    public String getProvider() {
        return provider;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public String getModel() {
        return model;
    }

    public String getApiKey() {
        return apiKey;
    }

    /**
     * 返回不含 api_key 的可公开配置，用于 API 响应。
     */
    public Map<String, Object> publicPayload() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("provider", provider);
        payload.put("base_url", baseUrl);
        payload.put("model", model);
        payload.put("has_api_key", true);
        return payload;
    }
}
