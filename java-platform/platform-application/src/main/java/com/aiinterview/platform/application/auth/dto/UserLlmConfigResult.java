package com.aiinterview.platform.application.auth.dto;

/**
 * 用户 LLM 配置查询结果（不含 api_key）。
 */
public class UserLlmConfigResult {

    private final String provider;
    private final String baseUrl;
    private final String model;
    private final boolean hasApiKey;

    public UserLlmConfigResult(String provider, String baseUrl, String model, boolean hasApiKey) {
        this.provider = provider;
        this.baseUrl = baseUrl;
        this.model = model;
        this.hasApiKey = hasApiKey;
    }

    public String getProvider() { return provider; }
    public String getBaseUrl() { return baseUrl; }
    public String getModel() { return model; }
    public boolean isHasApiKey() { return hasApiKey; }
}
