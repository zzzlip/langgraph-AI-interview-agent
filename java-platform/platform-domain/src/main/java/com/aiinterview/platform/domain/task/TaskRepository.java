package com.aiinterview.platform.domain.task;

import java.util.Optional;

/**
 * 任务聚合仓储接口。
 */
public interface TaskRepository {

    void save(Task task);

    Optional<Task> findById(String taskId);

    Optional<Task> findByTraceId(String traceId);

    PageResult<Task> findByUser(TaskQueryCriteria criteria);

    PageResult<Task> findFailed(FailedTaskQueryCriteria criteria);

    boolean updateByIdAndVersion(Task task, long expectedVersion);
}
