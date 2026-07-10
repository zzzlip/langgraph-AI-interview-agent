package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskPageResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * 任务分页响应。
 */
public record TaskPageResponse(
        List<TaskListItemResponse> items,
        long total,
        int page,
        @JsonProperty("page_size") int pageSize
) {
    public static TaskPageResponse from(TaskPageResult result) {
        return new TaskPageResponse(
                result.items().stream().map(TaskListItemResponse::from).toList(),
                result.total(),
                result.page(),
                result.pageSize()
        );
    }
}
