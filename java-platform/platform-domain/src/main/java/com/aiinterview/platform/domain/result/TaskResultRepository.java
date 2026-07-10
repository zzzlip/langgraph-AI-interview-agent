package com.aiinterview.platform.domain.result;

import java.util.List;
import java.util.Optional;

/**
 * 任务结果仓储接口。
 */
public interface TaskResultRepository {

    void save(TaskResult result);

    Optional<TaskResult> findById(String resultId);

    Optional<TaskResult> findByTaskIdAndAttemptNo(String taskId, int attemptNo);

    List<TaskResult> findByTaskId(String taskId);
}
