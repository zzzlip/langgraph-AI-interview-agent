package com.aiinterview.platform.application.file.dto;

import java.util.List;

/**
 * 任务关联文件分页结果。
 */
public record FileTaskFilePageResult(
        List<FileTaskFileItem> items,
        long total,
        int page,
        int pageSize
) {
}
