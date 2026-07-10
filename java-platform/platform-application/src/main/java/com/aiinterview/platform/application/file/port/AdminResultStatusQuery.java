package com.aiinterview.platform.application.file.port;

/**
 * 管理员结果状态查询条件。
 */
public record AdminResultStatusQuery(
        String taskId,
        String resultType,
        Integer attemptNo,
        int page,
        int pageSize
) {
}
