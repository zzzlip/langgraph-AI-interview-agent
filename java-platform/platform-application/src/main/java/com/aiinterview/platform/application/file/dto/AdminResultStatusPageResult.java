package com.aiinterview.platform.application.file.dto;

import java.util.List;

/**
 * 管理员结果状态分页结果。
 */
public record AdminResultStatusPageResult(
        List<AdminResultStatusItem> items,
        long total,
        int page,
        int pageSize
) {
    public AdminResultStatusPageResult {
        items = List.copyOf(items);
    }
}
