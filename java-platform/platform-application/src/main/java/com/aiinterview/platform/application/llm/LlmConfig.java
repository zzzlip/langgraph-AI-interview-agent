package com.aiinterview.platform.application.llm;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 用户任务级 LLM 配置。
 */
public record LlmConfig(
        String provider,
        String baseUrl,
        String model,
        String apiKey
) {
    public LlmConfig {
        Objects.requireNonNull(provider, "provider 不能为空");
        Objects.requireNonNull(baseUrl, "baseUrl 不能为空");
        Objects.requireNonNull(model, "model 不能为空");
        Objects.requireNonNull(apiKey, "apiKey 不能为空");
    }

    public Map<String, Object> privatePayload() {
        Map<String, Object> payload = publicPayload();
        payload.put("api_key", apiKey);
        return payload;
    }

    public Map<String, Object> publicPayload() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("provider", provider);
        payload.put("base_url", baseUrl);
        payload.put("model", model);
        return payload;
    }
}
