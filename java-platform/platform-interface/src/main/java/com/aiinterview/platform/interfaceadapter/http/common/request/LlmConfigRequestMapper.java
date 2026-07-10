package com.aiinterview.platform.interfaceadapter.http.common.request;

import com.aiinterview.platform.application.auth.UserLlmConfigApplicationService;
import com.aiinterview.platform.application.llm.LlmConfig;

/**
 * LLM 配置请求转换。
 */
public final class LlmConfigRequestMapper {

    private LlmConfigRequestMapper() {
    }

    public static LlmConfig toCommandConfig(
            String accountId,
            LlmConfigRequest request,
            UserLlmConfigApplicationService userLlmConfigApplicationService
    ) {
        if (request == null) {
            return null;
        }
        return userLlmConfigApplicationService.resolveTaskLlmConfig(
                accountId,
                request.getProvider(),
                request.getBaseUrl(),
                request.getModel(),
                request.getApiKey()
        );
    }

    public static String publicHashPart(LlmConfig config) {
        if (config == null) {
            return "";
        }
        // 幂等 hash 只能纳入公开配置，避免 api_key 进入日志、事件或持久化摘要。
        return config.provider() + ":" + config.baseUrl() + ":" + config.model();
    }
}
