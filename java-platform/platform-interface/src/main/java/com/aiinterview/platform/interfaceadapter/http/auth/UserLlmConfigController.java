package com.aiinterview.platform.interfaceadapter.http.auth;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.auth.UserLlmConfigApplicationService;
import com.aiinterview.platform.application.auth.command.SaveLlmConfigCommand;
import com.aiinterview.platform.application.auth.dto.UserLlmConfigResult;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.http.auth.request.SaveLlmConfigRequest;
import com.aiinterview.platform.interfaceadapter.http.auth.response.UserLlmConfigResponse;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import java.util.Optional;
import org.springframework.http.CacheControl;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 用户 LLM 配置接口。
 */
@RestController
@RequestMapping("/api/v1/user")
public class UserLlmConfigController {

    private final UserLlmConfigApplicationService userLlmConfigApplicationService;
    private final SessionManager sessionManager;

    public UserLlmConfigController(
            UserLlmConfigApplicationService userLlmConfigApplicationService,
            SessionManager sessionManager
    ) {
        this.userLlmConfigApplicationService = userLlmConfigApplicationService;
        this.sessionManager = sessionManager;
    }

    @PutMapping("/llm-config")
    public ResponseEntity<BaseResult<UserLlmConfigResponse>> saveLlmConfig(
            @Valid @RequestBody SaveLlmConfigRequest request,
            HttpSession session
    ) {
        CurrentSession currentSession = requireUserSession(session, sessionManager);
        UserLlmConfigResult result = userLlmConfigApplicationService.saveLlmConfig(
                currentSession.getAccountId(),
                new SaveLlmConfigCommand(
                        request.getProvider(),
                        request.getBaseUrl(),
                        request.getModel(),
                        request.getApiKey()
                )
        );
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(BaseResult.success(UserLlmConfigResponse.fromResult(result)));
    }

    @GetMapping("/llm-config")
    public ResponseEntity<BaseResult<UserLlmConfigResponse>> getLlmConfig(HttpSession session) {
        CurrentSession currentSession = requireUserSession(session, sessionManager);
        Optional<UserLlmConfigResult> result = userLlmConfigApplicationService.getLlmConfig(currentSession.getAccountId());
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .body(result.map(config -> BaseResult.success(UserLlmConfigResponse.fromResult(config)))
                        .orElseGet(() -> BaseResult.success(null)));
    }

}
