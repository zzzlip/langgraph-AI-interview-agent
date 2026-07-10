package com.aiinterview.platform.application.asyncmessage;

import com.aiinterview.platform.application.asyncmessage.port.RabbitMessagePublisher;
import com.aiinterview.platform.domain.asyncmessage.AsyncDeliveryStatus;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRecord;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterRepository;
import com.aiinterview.platform.domain.asyncmessage.DeadLetterSourceType;
import com.aiinterview.platform.domain.asyncmessage.OutboxEvent;
import com.aiinterview.platform.domain.asyncmessage.OutboxEventRepository;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Outbox 发布编排服务。
 */
public class OutboxPublishService {

    private static final Duration RETRY_BACKOFF = Duration.ofSeconds(30);

    private final OutboxEventRepository repository;
    private final DeadLetterRepository deadLetterRepository;
    private final RabbitMessagePublisher publisher;
    private final Clock clock;
    private final Supplier<String> idSupplier;

    public OutboxPublishService(OutboxEventRepository repository, DeadLetterRepository deadLetterRepository,
            RabbitMessagePublisher publisher, Clock clock, Supplier<String> idSupplier) {
        this.repository = Objects.requireNonNull(repository, "repository 不能为空");
        this.deadLetterRepository = Objects.requireNonNull(deadLetterRepository, "deadLetterRepository 不能为空");
        this.publisher = Objects.requireNonNull(publisher, "publisher 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
    }

    public int publishAvailable(int limit) {
        Instant now = Instant.now(clock);
        int processed = 0;
        for (OutboxEvent event : repository.findPublishable(now, limit)) {
            if (publishOne(event, now)) {
                processed++;
            }
        }
        return processed;
    }

    public int recoverStaleDelivering(Instant staleBefore, int limit) {
        Instant now = Instant.now(clock);
        int recovered = 0;
        for (OutboxEvent event : repository.findStaleDelivering(staleBefore, limit)) {
            OutboxEvent failed = event.markFailed("stale delivering recovered", now.plus(RETRY_BACKOFF), now);
            if (repository.updateByIdAndStatus(failed, AsyncDeliveryStatus.DELIVERING)) {
                recovered++;
            }
        }
        return recovered;
    }

    private boolean publishOne(OutboxEvent event, Instant now) {
        OutboxEvent delivering;
        try {
            delivering = event.markDelivering(now);
        } catch (IllegalStateException ex) {
            return false;
        }
        if (!repository.updateByIdAndStatus(delivering, event.getStatus())) {
            return false;
        }

        try {
            publisher.publish(delivering);
            OutboxEvent delivered = delivering.markDelivered(now);
            repository.updateByIdAndStatus(delivered, AsyncDeliveryStatus.DELIVERING);
        } catch (RuntimeException ex) {
            OutboxEvent failed = toFailureState(delivering, ex, now);
            boolean updated = repository.updateByIdAndStatus(failed, AsyncDeliveryStatus.DELIVERING);
            if (updated && failed.getStatus() == AsyncDeliveryStatus.DEAD) {
                saveOutboxDeadLetter(failed, now);
            }
        }
        return true;
    }

    private OutboxEvent toFailureState(OutboxEvent delivering, RuntimeException ex, Instant now) {
        String error = sanitize(ex);
        if (delivering.getRetryCount() >= delivering.getMaxRetryCount()) {
            return delivering.markDead(error, now);
        }
        return delivering.markFailed(error, now.plus(RETRY_BACKOFF), now);
    }

    private void saveOutboxDeadLetter(OutboxEvent event, Instant now) {
        if (deadLetterRepository.findBySource(DeadLetterSourceType.OUTBOX, event.getId()).isPresent()) {
            return;
        }
        DeadLetterRecord record = new DeadLetterRecord(
                idSupplier.get(),
                DeadLetterSourceType.OUTBOX,
                event.getId(),
                event.getTaskId(),
                event.getTaskType(),
                event.getAttemptNo(),
                "outbox_publish",
                event.getLastError() == null ? "Outbox 投递达到最大重试次数" : event.getLastError(),
                false,
                false,
                event.getTraceId(),
                now,
                now
        );
        deadLetterRepository.save(record);
    }

    private static String sanitize(RuntimeException ex) {
        String message = ex.getMessage();
        if (message == null || message.isBlank()) {
            return "publish failed";
        }
        return message.length() > 500 ? message.substring(0, 500) : message;
    }
}
