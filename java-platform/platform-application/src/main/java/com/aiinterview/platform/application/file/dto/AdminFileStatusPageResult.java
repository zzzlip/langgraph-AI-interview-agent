package com.aiinterview.platform.application.file.dto;

import java.util.List;

/**
 * 管理员文件状态分页结果。
 */
public record AdminFileStatusPageResult(
        List<AdminFileStatusItem> items,
        long total,
        int page,
        int pageSize
) {
    public AdminFileStatusPageResult {
        items = List.copyOf(items);
    }
}
