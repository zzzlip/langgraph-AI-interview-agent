package com.aiinterview.platform.domain.task;

import java.util.List;
import java.util.Objects;

/**
 * 仓储分页结果。
 */
public record PageResult<T>(
        List<T> items,
        long total,
        int page,
        int pageSize
) {
    public PageResult {
        items = List.copyOf(Objects.requireNonNull(items, "items 不能为空"));
    }
}
