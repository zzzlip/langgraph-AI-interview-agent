package com.aiinterview.platform.interfaceadapter.http.auth.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Size;

/**
 * 保存 LLM 配置请求。
 */
public class SaveLlmConfigRequest {

    @Size(max = 50, message = "provider 长度不能超过 50")
    @JsonProperty("provider")
    private String provider;

    @Size(max = 200, message = "base_url 长度不能超过 200")
    @JsonProperty("base_url")
    private String baseUrl;

    @Size(max = 100, message = "model 长度不能超过 100")
    @JsonProperty("model")
    private String model;

    @Size(max = 256, message = "api_key 长度不能超过 256")
    @JsonProperty("api_key")
    private String apiKey;

    public String getProvider() { return provider; }
    public void setProvider(String provider) { this.provider = provider; }
    public String getBaseUrl() { return baseUrl; }
    public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    public String getApiKey() { return apiKey; }
    public void setApiKey(String apiKey) { this.apiKey = apiKey; }
}
