package com.aiinterview.platform.domain.task;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * 任务幂等记录仓储接口。
 */
public interface TaskIdempotencyRepository {

    void save(TaskIdempotencyRecord record);

    Optional<TaskIdempotencyRecord> findByUserIdAndOperationTypeAndIdempotencyKey(
            String userId,
            String operationType,
            String idempotencyKey
    );

    List<TaskIdempotencyRecord> findExpiredBefore(Instant expiresAt);
}
