package com.aiinterview.platform.domain.task;

import java.util.List;
import java.util.Optional;

/**
 * 任务执行尝试仓储接口。
 */
public interface TaskAttemptRepository {

    void save(TaskAttempt attempt);

    Optional<TaskAttempt> findById(String attemptId);

    Optional<TaskAttempt> findByTaskIdAndAttemptNo(String taskId, int attemptNo);

    List<TaskAttempt> findByTaskId(String taskId);

    void update(TaskAttempt attempt);
}
