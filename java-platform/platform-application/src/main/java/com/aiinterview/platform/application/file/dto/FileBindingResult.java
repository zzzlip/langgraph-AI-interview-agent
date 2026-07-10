package com.aiinterview.platform.application.file.dto;

/**
 * 文件绑定结果。
 */
public record FileBindingResult(
        String bindingId,
        String fileId,
        String taskId,
        String bindType,
        int attemptNo
) {
}
