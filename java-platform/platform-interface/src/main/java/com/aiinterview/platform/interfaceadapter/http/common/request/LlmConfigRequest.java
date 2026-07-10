package com.aiinterview.platform.interfaceadapter.http.common.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Size;

/**
 * 用户任务级 LLM 配置请求。
 */
public class LlmConfigRequest {

    @JsonProperty("provider")
    private String provider;

    @JsonProperty("base_url")
    @Size(max = 200, message = "llm_config.base_url 长度不能超过 200")
    private String baseUrl;

    @JsonProperty("model")
    @Size(max = 80, message = "llm_config.model 长度不能超过 80")
    private String model;

    @JsonProperty("api_key")
    @Size(max = 300, message = "llm_config.api_key 长度不能超过 300")
    private String apiKey;

    public String getProvider() {
        return provider;
    }

    public void setProvider(String provider) {
        this.provider = provider;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getApiKey() {
        return apiKey;
    }

    public void setApiKey(String apiKey) {
        this.apiKey = apiKey;
    }
}
