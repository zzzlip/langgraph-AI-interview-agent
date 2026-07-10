package com.aiinterview.platform.application.resumeevaluation.port;

import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationCreatedResult;

/**
 * 简历评估任务创建端口。
 */
public interface ResumeEvaluationTaskPort {

    ResumeEvaluationCreatedResult createResumeEvaluationTask(ResumeEvaluationTaskCreation creation);
}
