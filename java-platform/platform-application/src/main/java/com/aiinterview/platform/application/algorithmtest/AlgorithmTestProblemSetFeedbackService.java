package com.aiinterview.platform.application.algorithmtest;

import com.aiinterview.platform.application.algorithmtest.command.AlgorithmProblemSetFeedbackCommand;
import com.aiinterview.platform.domain.result.ResultType;
import com.aiinterview.platform.domain.result.TaskResult;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.aiinterview.platform.domain.task.Task;
import com.aiinterview.platform.domain.task.TaskEvent;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskEventType;
import com.aiinterview.platform.domain.task.TaskOperatorType;
import com.aiinterview.platform.domain.task.TaskRepository;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.aiinterview.platform.domain.task.TaskType;
import java.time.Clock;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import org.springframework.transaction.annotation.Transactional;

/**
 * 算法测试题单中间反馈处理服务。
 */
public class AlgorithmTestProblemSetFeedbackService {

    private static final int TASK_NOT_FOUND_CODE = 71003;
    private static final int ATTEMPT_MISMATCH_CODE = 71011;
    private static final int VERSION_CONFLICT_CODE = 71009;
    private static final int TASK_TYPE_INVALID_CODE = 76005;
    private static final int TASK_STATUS_INVALID_CODE = 76006;
    private static final int PROBLEM_SET_INVALID_CODE = 76014;
    private static final String PROBLEM_SET_EVENT_TYPE = "algorithm_test.problem_set.generated";
    private static final String TASK_TYPE = "algorithm_test";
    private static final String WAITING_STATUS = "waiting_user_input";
    private static final int PROBLEM_COUNT = 5;

    private final TaskRepository taskRepository;
    private final TaskEventRepository eventRepository;
    private final TaskResultRepository taskResultRepository;
    private final Clock clock;
    private final Supplier<String> idSupplier;

    public AlgorithmTestProblemSetFeedbackService(
            TaskRepository taskRepository,
            TaskEventRepository eventRepository,
            TaskResultRepository taskResultRepository,
            Clock clock,
            Supplier<String> idSupplier
    ) {
        this.taskRepository = Objects.requireNonNull(taskRepository, "taskRepository 不能为空");
        this.eventRepository = Objects.requireNonNull(eventRepository, "eventRepository 不能为空");
        this.taskResultRepository = Objects.requireNonNull(taskResultRepository, "taskResultRepository 不能为空");
        this.clock = Objects.requireNonNull(clock, "clock 不能为空");
        this.idSupplier = Objects.requireNonNull(idSupplier, "idSupplier 不能为空");
    }

    @Transactional
    public boolean handleProblemSet(AlgorithmProblemSetFeedbackCommand command) {
        validateEnvelope(command);
        Task task = taskRepository.findById(command.taskId())
                .orElseThrow(() -> new AlgorithmTestException(TASK_NOT_FOUND_CODE, "任务不存在"));
        if (task.getTaskType() != TaskType.ALGORITHM_TEST) {
            throw new AlgorithmTestException(TASK_TYPE_INVALID_CODE, "只能处理算法测试题单反馈");
        }
        if (command.attemptNo() < task.getCurrentAttempt()) {
            appendIgnored(task, command, "过期题单反馈被忽略");
            return true;
        }
        if (command.attemptNo() > task.getCurrentAttempt()) {
            throw new AlgorithmTestException(ATTEMPT_MISMATCH_CODE, "Worker 反馈 attempt 不匹配");
        }
        if (!task.getTraceId().equals(command.traceId())) {
            throw new AlgorithmTestException(ATTEMPT_MISMATCH_CODE, "Worker 反馈 trace_id 不匹配");
        }
        if (task.getStatus().isFinal()) {
            appendIgnored(task, command, "终态任务忽略题单反馈");
            return true;
        }
        if (task.getStatus() != TaskStatus.RUNNING) {
            throw new AlgorithmTestException(TASK_STATUS_INVALID_CODE, "任务不在题单生成运行状态");
        }
        validateResultSummary(command.resultSummary());

        TaskStatus fromStatus = task.getStatus();
        long expectedVersion = task.getVersion();
        Instant now = Instant.now(clock);
        taskResultRepository.save(TaskResult.createCompleted(
                idSupplier.get(),
                task.getId(),
                command.attemptNo(),
                ResultType.ALGORITHM_TEST,
                command.resultSummary(),
                now
        ));
        task.markWaitingUserInput(command.stage(), command.message(), now);
        if (!taskRepository.updateByIdAndVersion(task, expectedVersion)) {
            throw new AlgorithmTestException(VERSION_CONFLICT_CODE, "任务版本冲突");
        }
        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                command.attemptNo(),
                TaskEventType.TASK_WAITING_USER_INPUT,
                fromStatus,
                task.getStatus(),
                command.stage(),
                command.progress(),
                command.message(),
                Map.of("schema_version", 1),
                TaskOperatorType.WORKER,
                "worker",
                command.traceId(),
                now,
                now
        ));
        return false;
    }

    private static void validateEnvelope(AlgorithmProblemSetFeedbackCommand command) {
        if (command == null
                || !PROBLEM_SET_EVENT_TYPE.equals(command.eventType())
                || !TASK_TYPE.equals(command.taskType())
                || command.attemptNo() < 1
                || !WAITING_STATUS.equals(command.status())
                || !WAITING_STATUS.equals(command.stage())
                || command.message() == null
                || command.message().isBlank()) {
            throw new AlgorithmTestException(PROBLEM_SET_INVALID_CODE, "算法测试题单反馈格式非法");
        }
    }

    private static void validateResultSummary(Map<String, Object> summary) {
        if (summary == null
                || !(summary.get("schema_version") instanceof Integer)
                || !"algorithm_test".equals(summary.get("result_type"))
                || !"problem_set".equals(summary.get("phase"))
                || !(summary.get("problems") instanceof List<?> problems)
                || problems.size() != PROBLEM_COUNT) {
            throw new AlgorithmTestException(PROBLEM_SET_INVALID_CODE, "算法测试题单数据不完整");
        }
        for (Object problem : problems) {
            if (!(problem instanceof Map<?, ?> problemMap)
                    || !(problemMap.get("contest_id") instanceof Number)
                    || !(problemMap.get("problem_index") instanceof String problemIndex)
                    || problemIndex.isBlank()
                    || !(problemMap.get("link") instanceof String link)
                    || !link.startsWith("https://codeforces.com/problemset/problem/")) {
                throw new AlgorithmTestException(PROBLEM_SET_INVALID_CODE, "算法测试题单判题键不完整");
            }
        }
    }

    private void appendIgnored(Task task, AlgorithmProblemSetFeedbackCommand command, String message) {
        Instant now = Instant.now(clock);
        eventRepository.save(new TaskEvent(
                idSupplier.get(),
                task.getId(),
                command.attemptNo(),
                TaskEventType.TASK_FEEDBACK_IGNORED,
                task.getStatus(),
                task.getStatus(),
                task.getStage(),
                task.getProgress(),
                message,
                Map.of("schema_version", 1),
                TaskOperatorType.WORKER,
                "worker",
                command.traceId(),
                now,
                now
        ));
    }
}
