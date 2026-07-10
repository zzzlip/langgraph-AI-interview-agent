package com.aiinterview.platform.application.task.port;

/**
 * 任务结果引用绑定端口。
 */
public interface TaskResultReferencePort {

    String bindResultReference(String taskId, int attemptNo, String resultPayload);

    /**
     * 命令对象入参用于 spec04 结果绑定；旧三参数方法在 spec03 迁移期继续兼容。
     */
    default String bindResultReference(BindResultReferenceCommand command) {
        return bindResultReference(command.taskId(), command.attemptNo(), command.summary());
    }
}
