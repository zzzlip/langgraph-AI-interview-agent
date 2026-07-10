package com.aiinterview.platform.domain.task;

/**
 * 普通用户任务列表查询条件。
 */
public record TaskQueryCriteria(
        String userId,
        TaskType taskType,
        TaskStatus status,
        int page,
        int pageSize
) {
}
