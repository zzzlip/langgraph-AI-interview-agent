package com.aiinterview.platform.interfaceadapter.http.file.response;

import com.aiinterview.platform.application.file.dto.FileTaskFilePageResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * 任务关联文件分页响应。
 */
public record TaskFilePageResponse(
        List<TaskFileItemResponse> items,
        long total,
        int page,
        @JsonProperty("page_size")
        int pageSize
) {

    public static TaskFilePageResponse from(FileTaskFilePageResult result) {
        return new TaskFilePageResponse(
                result.items().stream().map(TaskFileItemResponse::from).toList(),
                result.total(),
                result.page(),
                result.pageSize()
        );
    }
}
