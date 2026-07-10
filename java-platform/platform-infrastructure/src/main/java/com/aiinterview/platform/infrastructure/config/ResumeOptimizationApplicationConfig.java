package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.resumeoptimization.DefaultResumeOptimizationFileBindingPort;
import com.aiinterview.platform.application.resumeoptimization.DefaultResumeOptimizationTaskPort;
import com.aiinterview.platform.application.resumeoptimization.ResumeOptimizationApplicationService;
import com.aiinterview.platform.application.resumeoptimization.port.ResumeOptimizationFileBindingPort;
import com.aiinterview.platform.application.resumeoptimization.port.ResumeOptimizationTaskPort;
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
 * 简历优化应用服务装配。
 */
@Configuration
public class ResumeOptimizationApplicationConfig {

    @Bean
    public ResumeOptimizationFileBindingPort resumeOptimizationFileBindingPort(FileRepository fileRepository) {
        return new DefaultResumeOptimizationFileBindingPort(fileRepository);
    }

    @Bean
    public ResumeOptimizationTaskPort resumeOptimizationTaskPort(
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
        return new DefaultResumeOptimizationTaskPort(
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
    public ResumeOptimizationApplicationService resumeOptimizationApplicationService(
            ResumeOptimizationFileBindingPort fileBindingPort,
            ResumeOptimizationTaskPort taskPort,
            ObjectMapper objectMapper
    ) {
        return new ResumeOptimizationApplicationService(fileBindingPort, taskPort, objectMapper);
    }
}
