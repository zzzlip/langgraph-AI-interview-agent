package com.aiinterview.platform.interfaceadapter.http.mockinterview.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.aiinterview.platform.interfaceadapter.http.common.request.LlmConfigRequest;

/**
 * 创建模拟面试请求。
 */
public class CreateMockInterviewRequest {

    @JsonProperty("resume_file_id")
    private String resumeFileId;

    @JsonProperty("target_job")
    private TargetJob targetJob;

    @JsonProperty("question_count")
    private Integer questionCount;

    @JsonProperty("max_follow_up_rounds")
    private Integer maxFollowUpRounds;

    @JsonProperty("llm_config")
    private LlmConfigRequest llmConfig;

    public String getResumeFileId() {
        return resumeFileId;
    }

    public void setResumeFileId(String resumeFileId) {
        this.resumeFileId = resumeFileId;
    }

    public TargetJob getTargetJob() {
        return targetJob;
    }

    public void setTargetJob(TargetJob targetJob) {
        this.targetJob = targetJob;
    }

    public Integer getQuestionCount() {
        return questionCount;
    }

    public void setQuestionCount(Integer questionCount) {
        this.questionCount = questionCount;
    }

    public Integer getMaxFollowUpRounds() {
        return maxFollowUpRounds;
    }

    public void setMaxFollowUpRounds(Integer maxFollowUpRounds) {
        this.maxFollowUpRounds = maxFollowUpRounds;
    }

    public LlmConfigRequest getLlmConfig() {
        return llmConfig;
    }

    public void setLlmConfig(LlmConfigRequest llmConfig) {
        this.llmConfig = llmConfig;
    }

    /**
     * 目标岗位输入。
     */
    public static class TargetJob {

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
