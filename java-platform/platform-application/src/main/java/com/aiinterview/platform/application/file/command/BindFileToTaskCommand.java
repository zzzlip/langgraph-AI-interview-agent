package com.aiinterview.platform.application.file.command;

/**
 * 文件绑定任务命令。
 */
public record BindFileToTaskCommand(
        String currentUserId,
        String taskId,
        String fileId,
        String bindType,
        int attemptNo,
        String traceId
) {
}
