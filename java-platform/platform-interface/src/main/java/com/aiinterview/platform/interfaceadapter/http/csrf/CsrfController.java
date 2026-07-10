package com.aiinterview.platform.interfaceadapter.http.csrf;

import com.aiinterview.platform.application.csrf.CsrfApplicationService;
import com.aiinterview.platform.domain.common.BaseResult;
import com.aiinterview.platform.interfaceadapter.http.csrf.response.CsrfTokenResponse;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * CSRF Token 获取接口。
 */
@RestController
@RequestMapping("/api/v1")
public class CsrfController {

    private static final String CSRF_SESSION_MARKER = "CSRF_INITIALIZED";

    private final CsrfApplicationService csrfApplicationService;

    public CsrfController(CsrfApplicationService csrfApplicationService) {
        this.csrfApplicationService = csrfApplicationService;
    }

    @GetMapping("/csrf-token")
    public ResponseEntity<BaseResult<CsrfTokenResponse>> getCsrfToken(HttpSession session) {
        session.setAttribute(CSRF_SESSION_MARKER, Boolean.TRUE);
        String sessionId = session.getId();
        String token = csrfApplicationService.refreshToken(sessionId).getToken();
        BaseResult<CsrfTokenResponse> result = BaseResult.success(new CsrfTokenResponse(token));
        return ResponseEntity.ok()
                .header("Cache-Control", "no-cache, no-store, must-revalidate")
                .header("Pragma", "no-cache")
                .body(result);
    }
}
