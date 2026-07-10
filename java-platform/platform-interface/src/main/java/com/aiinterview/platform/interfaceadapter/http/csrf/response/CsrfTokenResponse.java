package com.aiinterview.platform.interfaceadapter.http.csrf.response;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * CSRF Token 响应。
 */
public class CsrfTokenResponse {

    @JsonProperty("csrf_token")
    private final String csrfToken;

    public CsrfTokenResponse(String csrfToken) {
        this.csrfToken = csrfToken;
    }

    public String getCsrfToken() {
        return csrfToken;
    }
}
