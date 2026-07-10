package com.aiinterview.platform.application.resumeoptimization.port;

import com.aiinterview.platform.application.resumeoptimization.dto.ResumeOptimizationInputFile;

/**
 * 简历优化输入文件校验端口。
 */
public interface ResumeOptimizationFileBindingPort {

    ResumeOptimizationInputFile validateInputResume(String userId, String resumeFileId);

    ResumeOptimizationInputFile validateOptionalAvatar(String userId, String avatarFileId);
}
