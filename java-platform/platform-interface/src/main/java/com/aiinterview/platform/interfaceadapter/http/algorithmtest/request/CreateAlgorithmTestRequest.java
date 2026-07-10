package com.aiinterview.platform.interfaceadapter.http.algorithmtest.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.aiinterview.platform.interfaceadapter.http.common.request.LlmConfigRequest;

/**
 * 创建算法测试任务请求。
 */
public class CreateAlgorithmTestRequest {

    @JsonProperty("target_job")
    private TargetJobRequest targetJob;

    @JsonProperty("codeforces_handle")
    private String codeforcesHandle;

    @JsonProperty("llm_config")
    private LlmConfigRequest llmConfig;

    public TargetJobRequest getTargetJob() {
        return targetJob;
    }

    public void setTargetJob(TargetJobRequest targetJob) {
        this.targetJob = targetJob;
    }

    public String getCodeforcesHandle() {
        return codeforcesHandle;
    }

    public void setCodeforcesHandle(String codeforcesHandle) {
        this.codeforcesHandle = codeforcesHandle;
    }

    public LlmConfigRequest getLlmConfig() {
        return llmConfig;
    }

    public void setLlmConfig(LlmConfigRequest llmConfig) {
        this.llmConfig = llmConfig;
    }

    /**
     * 目标岗位请求。
     */
    public static class TargetJobRequest {

        private String title;

        private String description;

        public String getTitle() {
            return title;
        }

        public void setTitle(String title) {
            this.title = title;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }
    }
}
