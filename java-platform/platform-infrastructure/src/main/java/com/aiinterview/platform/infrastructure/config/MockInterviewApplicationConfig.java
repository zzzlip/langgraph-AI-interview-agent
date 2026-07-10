package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.mockinterview.DefaultMockInterviewTaskPort;
import com.aiinterview.platform.application.mockinterview.DefaultMockInterviewFileBindingPort;
import com.aiinterview.platform.application.mockinterview.MockInterviewApplicationService;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewFileBindingPort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewMemoryCachePort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewResultPort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewTaskPort;
import com.aiinterview.platform.application.file.FileApplicationService;
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
 * 模拟面试应用服务装配。
 */
@Configuration
public class MockInterviewApplicationConfig {

    @Bean
    public MockInterviewTaskPort mockInterviewTaskPort(
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
        return new DefaultMockInterviewTaskPort(
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
    public MockInterviewApplicationService mockInterviewApplicationService(
            MockInterviewTaskPort taskPort,
            ObjectMapper objectMapper,
            TaskRepository taskRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            MockInterviewResultPort resultPort,
            MockInterviewFileBindingPort fileBindingPort,
            FileApplicationService fileApplicationService,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier,
            MockInterviewMemoryCachePort memoryCachePort
    ) {
        return new MockInterviewApplicationService(
                taskPort,
                objectMapper,
                taskRepository,
                eventRepository,
                idempotencyRepository,
                resultPort,
                fileBindingPort,
                fileApplicationService,
                outboxPort,
                auditWritePort,
                clock,
                idSupplier,
                memoryCachePort
        );
    }

    @Bean
    public MockInterviewFileBindingPort mockInterviewFileBindingPort(FileRepository fileRepository) {
        return new DefaultMockInterviewFileBindingPort(fileRepository);
    }
}
