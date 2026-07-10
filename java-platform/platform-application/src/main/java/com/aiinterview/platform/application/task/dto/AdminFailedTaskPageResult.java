package com.aiinterview.platform.application.task.dto;

import java.util.List;
import java.util.Objects;

/**
 * 管理员失败任务分页结果。
 */
public record AdminFailedTaskPageResult(
        List<AdminFailedTaskItem> items,
        long total,
        int page,
        int pageSize
) {
    public AdminFailedTaskPageResult {
        items = List.copyOf(Objects.requireNonNull(items, "items 不能为空"));
    }
}
