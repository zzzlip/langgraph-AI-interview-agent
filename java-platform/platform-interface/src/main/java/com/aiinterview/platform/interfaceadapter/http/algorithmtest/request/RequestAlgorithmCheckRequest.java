package com.aiinterview.platform.interfaceadapter.http.algorithmtest.request;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 发起算法测试提交检查请求。
 */
public class RequestAlgorithmCheckRequest {

    @JsonProperty("force_refresh")
    private Boolean forceRefresh;

    public Boolean getForceRefresh() {
        return forceRefresh;
    }

    public void setForceRefresh(Boolean forceRefresh) {
        this.forceRefresh = forceRefresh;
    }
}
