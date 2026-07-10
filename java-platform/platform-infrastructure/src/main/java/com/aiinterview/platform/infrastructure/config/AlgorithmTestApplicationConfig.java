package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.algorithmtest.AlgorithmTestApplicationService;
import com.aiinterview.platform.application.algorithmtest.DefaultAlgorithmTestTaskPort;
import com.aiinterview.platform.application.algorithmtest.port.AlgorithmTestTaskPort;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.task.TaskAttemptRepository;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskIdempotencyRepository;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.util.function.Supplier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 算法测试应用服务装配。
 */
@Configuration
public class AlgorithmTestApplicationConfig {

    @Bean
    public AlgorithmTestTaskPort algorithmTestTaskPort(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            TaskResultRepository taskResultRepository,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier,
            ObjectMapper objectMapper
    ) {
        return new DefaultAlgorithmTestTaskPort(
                taskRepository,
                attemptRepository,
                eventRepository,
                idempotencyRepository,
                taskResultRepository,
                outboxPort,
                auditWritePort,
                clock,
                idSupplier,
                objectMapper
        );
    }

    @Bean
    public AlgorithmTestApplicationService algorithmTestApplicationService(
            AlgorithmTestTaskPort taskPort,
            ObjectMapper objectMapper
    ) {
        return new AlgorithmTestApplicationService(taskPort, objectMapper);
    }
}
