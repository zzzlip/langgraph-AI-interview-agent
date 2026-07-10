package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.resumeevaluation.DefaultResumeEvaluationFileBindingPort;
import com.aiinterview.platform.application.resumeevaluation.DefaultResumeEvaluationTaskPort;
import com.aiinterview.platform.application.resumeevaluation.ResumeEvaluationApplicationService;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationFileBindingPort;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationTaskPort;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.task.TaskAttemptRepository;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskIdempotencyRepository;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.util.function.Supplier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 简历评估应用服务装配。
 */
@Configuration
public class ResumeEvaluationApplicationConfig {

    @Bean
    public ResumeEvaluationFileBindingPort resumeEvaluationFileBindingPort(FileRepository fileRepository) {
        return new DefaultResumeEvaluationFileBindingPort(fileRepository);
    }

    @Bean
    public ResumeEvaluationTaskPort resumeEvaluationTaskPort(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            FileBindingRepository fileBindingRepository,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier,
            ObjectMapper objectMapper
    ) {
        return new DefaultResumeEvaluationTaskPort(
                taskRepository,
                attemptRepository,
                eventRepository,
                idempotencyRepository,
                fileBindingRepository,
                outboxPort,
                auditWritePort,
                clock,
                idSupplier,
                objectMapper
        );
    }

    @Bean
    public ResumeEvaluationApplicationService resumeEvaluationApplicationService(
            ResumeEvaluationFileBindingPort fileBindingPort,
            ResumeEvaluationTaskPort taskPort,
            ObjectMapper objectMapper
    ) {
        return new ResumeEvaluationApplicationService(fileBindingPort, taskPort, objectMapper);
    }
}
