package com.aiinterview.platform.application.resumeoptimization.port;

import com.aiinterview.platform.application.resumeoptimization.dto.ResumeOptimizationCreatedResult;

/**
 * 简历优化任务创建端口。
 */
public interface ResumeOptimizationTaskPort {

    ResumeOptimizationCreatedResult createResumeOptimizationTask(ResumeOptimizationTaskCreation creation);
}
