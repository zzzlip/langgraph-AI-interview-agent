package com.aiinterview.platform.application.result;

import com.aiinterview.platform.application.task.port.BindResultReferenceCommand;
import com.aiinterview.platform.application.task.port.TaskResultReferencePort;
import java.util.Objects;

/**
 * spec04 任务结果引用 Port 适配。
 */
public class Spec04TaskResultReferencePort implements TaskResultReferencePort {

    private final ResultApplicationService resultApplicationService;

    public Spec04TaskResultReferencePort(ResultApplicationService resultApplicationService) {
        this.resultApplicationService = Objects.requireNonNull(resultApplicationService, "resultApplicationService 不能为空");
    }

    @Override
    public String bindResultReference(BindResultReferenceCommand command) {
        return resultApplicationService.saveCompletedResult(command);
    }

    @Override
    public String bindResultReference(String taskId, int attemptNo, String resultPayload) {
        throw new UnsupportedOperationException("spec04 结果绑定必须使用命令对象");
    }
}
