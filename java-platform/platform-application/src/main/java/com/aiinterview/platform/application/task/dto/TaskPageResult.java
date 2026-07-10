package com.aiinterview.platform.application.task.dto;

import java.util.List;
import java.util.Objects;

/**
 * 任务分页结果。
 */
public record TaskPageResult(
        List<TaskListItem> items,
        long total,
        int page,
        int pageSize
) {
    public TaskPageResult {
        items = List.copyOf(Objects.requireNonNull(items, "items 不能为空"));
    }
}
