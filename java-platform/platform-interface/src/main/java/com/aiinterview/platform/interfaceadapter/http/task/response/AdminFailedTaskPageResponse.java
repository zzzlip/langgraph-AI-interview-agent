package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.AdminFailedTaskPageResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * 管理员失败任务分页响应。
 */
public record AdminFailedTaskPageResponse(
        List<AdminFailedTaskItemResponse> items,
        long total,
        int page,
        @JsonProperty("page_size") int pageSize
) {
    public static AdminFailedTaskPageResponse from(AdminFailedTaskPageResult result) {
        return new AdminFailedTaskPageResponse(
                result.items().stream().map(AdminFailedTaskItemResponse::from).toList(),
                result.total(),
                result.page(),
                result.pageSize()
        );
    }
}
