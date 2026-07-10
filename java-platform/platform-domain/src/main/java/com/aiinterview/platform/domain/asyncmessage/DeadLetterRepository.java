package com.aiinterview.platform.domain.asyncmessage;

import java.util.List;
import java.util.Optional;

/**
 * 死信记录仓储接口。
 */
public interface DeadLetterRepository {

    void save(DeadLetterRecord record);

    void update(DeadLetterRecord record);

    Optional<DeadLetterRecord> findById(String id);

    Optional<DeadLetterRecord> findBySource(DeadLetterSourceType sourceType, String sourceId);

    List<DeadLetterRecord> findByTaskId(String taskId);

    List<DeadLetterRecord> findPage(DeadLetterSourceType sourceType, Boolean handled, int page, int pageSize);

    long count(DeadLetterSourceType sourceType, Boolean handled);
}
