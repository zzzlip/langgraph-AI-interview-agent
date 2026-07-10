package com.aiinterview.platform.interfaceadapter.http.file.response;

import com.aiinterview.platform.application.file.dto.AdminResultStatusPageResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * 管理员结果状态分页响应。
 */
public record AdminResultStatusPageResponse(
        List<AdminResultStatusItemResponse> items,
        long total,
        int page,
        @JsonProperty("page_size") int pageSize
) {
    public static AdminResultStatusPageResponse from(AdminResultStatusPageResult result) {
        return new AdminResultStatusPageResponse(
                result.items().stream().map(AdminResultStatusItemResponse::from).toList(),
                result.total(),
                result.page(),
                result.pageSize()
        );
    }
}
