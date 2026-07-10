package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.domain.task.TaskEventType;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.aiinterview.platform.domain.task.TaskType;

/**
 * 任务枚举持久化编码转换。
 */
final class TaskEnumMapper {

    private TaskEnumMapper() {
    }

    static TaskStatus statusFromCode(String code) {
        for (TaskStatus status : TaskStatus.values()) {
            if (status.getCode().equals(code)) {
                return status;
            }
        }
        throw new IllegalArgumentException("未知任务状态: " + code);
    }

    static TaskType typeFromCode(String code) {
        for (TaskType type : TaskType.values()) {
            if (type.getCode().equals(code)) {
                return type;
            }
        }
        throw new IllegalArgumentException("未知任务类型: " + code);
    }

    static TaskEventType eventTypeFromCode(String code) {
        for (TaskEventType type : TaskEventType.values()) {
            if (type.getCode().equals(code)) {
                return type;
            }
        }
        throw new IllegalArgumentException("未知任务事件类型: " + code);
    }
}
