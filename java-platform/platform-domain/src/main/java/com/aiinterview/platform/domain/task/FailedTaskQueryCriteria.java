package com.aiinterview.platform.domain.task;

/**
 * 管理员失败任务查询条件。
 */
public record FailedTaskQueryCriteria(
        TaskType taskType,
        Boolean retryable,
        int page,
        int pageSize
) {
}
