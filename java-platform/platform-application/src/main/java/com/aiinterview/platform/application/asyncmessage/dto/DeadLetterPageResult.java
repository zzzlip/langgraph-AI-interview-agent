package com.aiinterview.platform.application.asyncmessage.dto;

import java.util.List;
import java.util.Objects;

/**
 * 管理员死信分页结果。
 */
public record DeadLetterPageResult(
        List<DeadLetterItemResult> items,
        long total,
        int page,
        int pageSize
) {
    public DeadLetterPageResult {
        items = List.copyOf(Objects.requireNonNull(items, "items 不能为空"));
    }
}
