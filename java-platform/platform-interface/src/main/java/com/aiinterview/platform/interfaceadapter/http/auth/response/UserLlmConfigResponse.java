package com.aiinterview.platform.interfaceadapter.http.auth.response;

import com.aiinterview.platform.application.auth.dto.UserLlmConfigResult;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 用户 LLM 配置响应（不含 api_key）。
 */
public class UserLlmConfigResponse {

    @JsonProperty("provider")
    private String provider;

    @JsonProperty("base_url")
    private String baseUrl;

    @JsonProperty("model")
    private String model;

    @JsonProperty("has_api_key")
    private boolean hasApiKey;

    public static UserLlmConfigResponse fromResult(UserLlmConfigResult result) {
        UserLlmConfigResponse response = new UserLlmConfigResponse();
        response.provider = result.getProvider();
        response.baseUrl = result.getBaseUrl();
        response.model = result.getModel();
        response.hasApiKey = result.isHasApiKey();
        return response;
    }

    public String getProvider() { return provider; }
    public void setProvider(String provider) { this.provider = provider; }
    public String getBaseUrl() { return baseUrl; }
    public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    public boolean isHasApiKey() { return hasApiKey; }
    public void setHasApiKey(boolean hasApiKey) { this.hasApiKey = hasApiKey; }
}
