package com.aiinterview.platform.application.auth.command;

/**
 * 保存用户 LLM 配置命令。
 */
public class SaveLlmConfigCommand {

    private final String provider;
    private final String baseUrl;
    private final String model;
    private final String apiKey;

    public SaveLlmConfigCommand(String provider, String baseUrl, String model, String apiKey) {
        this.provider = provider;
        this.baseUrl = baseUrl;
        this.model = model;
        this.apiKey = apiKey;
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
}
