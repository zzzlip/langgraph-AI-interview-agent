package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.asyncmessage.AsyncAdminQueryService;
import com.aiinterview.platform.application.asyncmessage.OutboxPublishService;
import com.aiinterview.platform.application.asyncmessage.SseTaskEventService;
import com.aiinterview.platform.application.asyncmessage.TaskCommandOutboxService;
import com.aiinterview.platform.application.asyncmessage.WorkerFeedbackConsumeService;
import com.aiinterview.platform.application.asyncmessage.WorkerFeedbackInboxService;
import com.aiinterview.platform.application.asyncmessage.port.RabbitMessagePublisher;
import com.aiinterview.platform.application.asyncmessage.port.SseConnectionRegistry;
import com.aiinterview.platform.application.asyncmessage.port.SseTaskEventPublisher;
import com.aiinterview.platform.application.algorithmtest.AlgorithmTestProblemSetFeedbackService;
import com.aiinterview.platform.application.mockinterview.MockInterviewFeedbackService;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewMemoryCachePort;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewResultPort;
import com.aiinterview.platform.application.task.TaskWorkerFeedbackService;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRepository;
import com.aiinterview.platform.domain.asyncmessage.InboxEventRepository;
import com.aiinterview.platform.domain.asyncmessage.OutboxEventRepository;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.infrastructure.asyncmessage.AlgorithmProblemSetFeedbackDispatcher;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.util.function.Supplier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

/**
 * 异步消息应用服务装配。
 */
@Configuration
public class AsyncMessageApplicationConfig {

    @Bean
    @Primary
    public TaskCommandOutboxPort spec05TaskCommandOutboxPort(
            OutboxEventRepository outboxEventRepository,
            Clock clock,
            Supplier<String> idSupplier,
            ObjectMapper objectMapper
    ) {
        return new TaskCommandOutboxService(outboxEventRepository, clock, idSupplier, objectMapper);
    }

    @Bean
    public OutboxPublishService outboxPublishService(
            OutboxEventRepository outboxEventRepository,
            DeadLetterRepository deadLetterRepository,
            RabbitMessagePublisher rabbitMessagePublisher,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        return new OutboxPublishService(outboxEventRepository, deadLetterRepository, rabbitMessagePublisher, clock,
                idSupplier);
    }

    @Bean
    public WorkerFeedbackInboxService workerFeedbackInboxService(
            InboxEventRepository inboxEventRepository,
            DeadLetterRepository deadLetterRepository,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        return new WorkerFeedbackInboxService(inboxEventRepository, deadLetterRepository, clock, idSupplier);
    }

    @Bean
    public WorkerFeedbackConsumeService workerFeedbackConsumeService(
            WorkerFeedbackInboxService inboxService,
            InboxEventRepository inboxEventRepository,
            DeadLetterRepository deadLetterRepository,
            TaskWorkerFeedbackService taskWorkerFeedbackService,
            SseTaskEventPublisher sseTaskEventPublisher,
            ObjectMapper objectMapper,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        return new WorkerFeedbackConsumeService(inboxService, inboxEventRepository, deadLetterRepository,
                taskWorkerFeedbackService, sseTaskEventPublisher, objectMapper, clock, idSupplier);
    }

    @Bean
    public AlgorithmTestProblemSetFeedbackService algorithmTestProblemSetFeedbackService(
            TaskRepository taskRepository,
            TaskEventRepository taskEventRepository,
            TaskResultRepository taskResultRepository,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        return new AlgorithmTestProblemSetFeedbackService(
                taskRepository,
                taskEventRepository,
                taskResultRepository,
                clock,
                idSupplier
        );
    }

    @Bean
    public AlgorithmProblemSetFeedbackDispatcher algorithmProblemSetFeedbackDispatcher(
            WorkerFeedbackConsumeService workerFeedbackConsumeService,
            WorkerFeedbackInboxService inboxService,
            InboxEventRepository inboxEventRepository,
            AlgorithmTestProblemSetFeedbackService problemSetFeedbackService,
            MockInterviewFeedbackService mockInterviewFeedbackService,
            SseTaskEventPublisher sseTaskEventPublisher,
            ObjectMapper objectMapper,
            Clock clock
    ) {
        return new AlgorithmProblemSetFeedbackDispatcher(
                workerFeedbackConsumeService,
                inboxService,
                inboxEventRepository,
                problemSetFeedbackService,
                mockInterviewFeedbackService,
                sseTaskEventPublisher,
                objectMapper,
                clock
        );
    }

    @Bean
    public MockInterviewFeedbackService mockInterviewFeedbackService(
            TaskRepository taskRepository,
            TaskEventRepository taskEventRepository,
            MockInterviewResultPort mockInterviewResultPort,
            TaskCommandOutboxPort outboxPort,
            Clock clock,
            Supplier<String> idSupplier,
            MockInterviewMemoryCachePort memoryCachePort
    ) {
        return new MockInterviewFeedbackService(
                taskRepository,
                taskEventRepository,
                mockInterviewResultPort,
                outboxPort,
                clock,
                idSupplier,
                memoryCachePort
        );
    }

    @Bean
    @Primary
    public SseTaskEventService sseTaskEventService(
            TaskRepository taskRepository,
            TaskEventRepository taskEventRepository,
            SseConnectionRegistry sseConnectionRegistry,
            AuditWritePort auditWritePort,
            PlatformProperties platformProperties,
            Clock clock
    ) {
        PlatformProperties.AsyncMessage asyncMessage = platformProperties.getAsyncMessage();
        return new SseTaskEventService(taskRepository, taskEventRepository, sseConnectionRegistry, auditWritePort, clock,
                asyncMessage.getMaxConnectionsPerUser(), asyncMessage.getMaxConnectionsPerTask());
    }

    @Bean
    public AsyncAdminQueryService asyncAdminQueryService(
            TaskRepository taskRepository,
            OutboxEventRepository outboxEventRepository,
            InboxEventRepository inboxEventRepository,
            DeadLetterRepository deadLetterRepository,
            AuditWritePort auditWritePort,
            Clock clock
    ) {
        return new AsyncAdminQueryService(taskRepository, outboxEventRepository, inboxEventRepository,
                deadLetterRepository, auditWritePort, clock);
    }
}
