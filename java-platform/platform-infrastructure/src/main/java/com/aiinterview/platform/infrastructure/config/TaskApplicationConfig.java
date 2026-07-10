package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.result.ResultQueryService;
import com.aiinterview.platform.application.task.TaskAdminApplicationService;
import com.aiinterview.platform.application.task.TaskApplicationService;
import com.aiinterview.platform.application.task.TaskQueryService;
import com.aiinterview.platform.application.task.TaskWorkerFeedbackService;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.ResourceOwnershipPort;
import com.aiinterview.platform.application.task.port.TaskCommandOutboxPort;
import com.aiinterview.platform.application.task.port.TaskResultReferencePort;
import com.aiinterview.platform.domain.task.TaskAttemptRepository;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskIdempotencyRepository;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.util.function.Supplier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 任务中心应用服务装配。
 */
@Configuration
public class TaskApplicationConfig {

    @Bean
    public TaskQueryService taskQueryService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            AuditWritePort auditWritePort,
            ResultQueryService resultQueryService
    ) {
        return new TaskQueryService(taskRepository, attemptRepository, eventRepository, auditWritePort, resultQueryService);
    }

    @Bean
    public TaskApplicationService taskApplicationService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            TaskCommandOutboxPort outboxPort,
            ResourceOwnershipPort resourceOwnershipPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        return new TaskApplicationService(
                taskRepository,
                attemptRepository,
                eventRepository,
                idempotencyRepository,
                outboxPort,
                resourceOwnershipPort,
                auditWritePort,
                clock,
                idSupplier
        );
    }

    @Bean
    public TaskAdminApplicationService taskAdminApplicationService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskIdempotencyRepository idempotencyRepository,
            TaskCommandOutboxPort outboxPort,
            AuditWritePort auditWritePort,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        return new TaskAdminApplicationService(
                taskRepository,
                attemptRepository,
                eventRepository,
                idempotencyRepository,
                outboxPort,
                auditWritePort,
                clock,
                idSupplier
        );
    }

    @Bean
    public TaskWorkerFeedbackService taskWorkerFeedbackService(
            TaskRepository taskRepository,
            TaskAttemptRepository attemptRepository,
            TaskEventRepository eventRepository,
            TaskResultReferencePort resultReferencePort,
            Clock clock,
            Supplier<String> idSupplier,
            ObjectMapper objectMapper
    ) {
        return new TaskWorkerFeedbackService(
                taskRepository,
                attemptRepository,
                eventRepository,
                resultReferencePort,
                clock,
                idSupplier,
                objectMapper
        );
    }

    @Bean
    @ConditionalOnMissingBean
    public ResourceOwnershipPort resourceOwnershipPort(TaskRepository taskRepository) {
        // 当前已接入 spec03 任务归属；其他资源类型保持 fail-closed。
        return (userId, resourceType, resourceId) -> {
            if ("task".equals(resourceType)) {
                return taskRepository.findById(resourceId)
                        .map(task -> task.getUserId().equals(userId))
                        .orElse(false);
            }
            return false;
        };
    }

    @Bean
    @ConditionalOnMissingBean
    public TaskCommandOutboxPort taskCommandOutboxPort() {
        // 兜底替身仅用于缺省测试上下文；正式链路由 spec05 的 @Primary Outbox 适配器覆盖。
        return (task, commandType) -> {
        };
    }

    @Bean
    @ConditionalOnMissingBean
    public TaskResultReferencePort taskResultReferencePort() {
        // 兜底替身仅用于缺省测试上下文；正式链路由 spec04 的 @Primary 结果适配器覆盖。
        return (taskId, attemptNo, resultPayload) -> "result:" + taskId + ":" + attemptNo;
    }

    @Bean
    @ConditionalOnMissingBean
    public AuditWritePort auditWritePort() {
        // spec02 审计枚举尚未覆盖全部任务事件，先保持端口可替换且不阻断任务链路。
        return (eventType, actorId, targetId, details) -> {
        };
    }
}
