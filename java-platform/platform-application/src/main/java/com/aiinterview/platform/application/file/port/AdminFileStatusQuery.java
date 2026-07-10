package com.aiinterview.platform.application.file.port;

/**
 * 管理员文件状态查询条件。
 */
public record AdminFileStatusQuery(
        String fileStatus,
        String filePurpose,
        String taskId,
        int page,
        int pageSize
) {
}
