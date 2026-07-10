package com.aiinterview.platform.interfaceadapter.http.resumeoptimization.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.aiinterview.platform.interfaceadapter.http.common.request.LlmConfigRequest;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

/**
 * 创建简历优化任务请求。
 */
public class CreateResumeOptimizationRequest {

    private static final String UUID_PATTERN =
            "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$";

    @JsonProperty("resume_file_id")
    @NotBlank(message = "resume_file_id 不能为空")
    @Pattern(regexp = UUID_PATTERN, message = "resume_file_id 格式非法")
    private String resumeFileId;

    @JsonProperty("avatar_file_id")
    @Pattern(regexp = UUID_PATTERN, message = "avatar_file_id 格式非法")
    private String avatarFileId;

    @Valid
    @NotNull(message = "target_job 不能为空")
    @JsonProperty("target_job")
    private TargetJobRequest targetJob;

    @Valid
    @JsonProperty("llm_config")
    private LlmConfigRequest llmConfig;

    public String getResumeFileId() {
        return resumeFileId;
    }

    public void setResumeFileId(String resumeFileId) {
        this.resumeFileId = resumeFileId;
    }

    public String getAvatarFileId() {
        return avatarFileId;
    }

    public void setAvatarFileId(String avatarFileId) {
        this.avatarFileId = avatarFileId;
    }

    public TargetJobRequest getTargetJob() {
        return targetJob;
    }

    public void setTargetJob(TargetJobRequest targetJob) {
        this.targetJob = targetJob;
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

        @NotBlank(message = "target_job.title 不能为空")
        @Size(max = 80, message = "target_job.title 长度不能超过 80")
        private String title;

        @NotBlank(message = "target_job.description 不能为空")
        @Size(min = 20, max = 4000, message = "target_job.description 长度必须为 20 到 4000")
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
