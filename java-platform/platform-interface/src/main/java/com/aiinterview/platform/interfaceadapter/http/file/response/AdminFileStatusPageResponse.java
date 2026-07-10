package com.aiinterview.platform.interfaceadapter.http.file.response;

import com.aiinterview.platform.application.file.dto.AdminFileStatusPageResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * 管理员文件状态分页响应。
 */
public record AdminFileStatusPageResponse(
        List<AdminFileStatusItemResponse> items,
        long total,
        int page,
        @JsonProperty("page_size") int pageSize
) {
    public static AdminFileStatusPageResponse from(AdminFileStatusPageResult result) {
        return new AdminFileStatusPageResponse(
                result.items().stream().map(AdminFileStatusItemResponse::from).toList(),
                result.total(),
                result.page(),
                result.pageSize()
        );
    }
}
