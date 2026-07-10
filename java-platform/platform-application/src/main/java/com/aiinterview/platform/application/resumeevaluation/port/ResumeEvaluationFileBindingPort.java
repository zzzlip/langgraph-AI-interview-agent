package com.aiinterview.platform.application.resumeevaluation.port;

import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationInputFile;

/**
 * 简历评估输入文件校验端口。
 */
public interface ResumeEvaluationFileBindingPort {

    ResumeEvaluationInputFile validateInputResume(String userId, String resumeFileId);
}
