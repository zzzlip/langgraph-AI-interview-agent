package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.domain.task.TaskEvent;
import com.aiinterview.platform.domain.task.TaskEventRepository;
import com.aiinterview.platform.domain.task.TaskOperatorType;
import com.aiinterview.platform.domain.task.TaskStatus;
import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskEventEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * 任务事件仓储基础设施实现。
 */
@Component
public class TaskEventRepositoryImpl implements TaskEventRepository {

    private static final TypeReference<Map<String, Object>> PAYLOAD_TYPE = new TypeReference<>() {
    };

    private final TaskEventMapper taskEventMapper;
    private final ObjectMapper objectMapper;

    public TaskEventRepositoryImpl(TaskEventMapper taskEventMapper, ObjectMapper objectMapper) {
        this.taskEventMapper = taskEventMapper;
        this.objectMapper = objectMapper;
    }

    @Override
    public void save(TaskEvent event) {
        taskEventMapper.insert(toEntity(event));
    }

    @Override
    public Optional<TaskEvent> findById(String eventId) {
        return Optional.ofNullable(taskEventMapper.selectById(eventId)).map(this::toDomain);
    }

    @Override
    public List<TaskEvent> findByTaskIdOrderByOccurredAt(String taskId) {
        return taskEventMapper.selectList(new LambdaQueryWrapper<TaskEventEntity>()
                .eq(TaskEventEntity::getTaskId, taskId)
                .orderByAsc(TaskEventEntity::getOccurredAt))
                .stream().map(this::toDomain).toList();
    }

    @Override
    public List<TaskEvent> findByTaskIdAndOccurredAtBetween(String taskId, Instant startInclusive, Instant endExclusive) {
        return taskEventMapper.selectList(new LambdaQueryWrapper<TaskEventEntity>()
                .eq(TaskEventEntity::getTaskId, taskId)
                .ge(TaskEventEntity::getOccurredAt, startInclusive)
                .lt(TaskEventEntity::getOccurredAt, endExclusive)
                .orderByAsc(TaskEventEntity::getOccurredAt))
                .stream().map(this::toDomain).toList();
    }

    @Override
    public List<TaskEvent> findByTraceId(String traceId) {
        return taskEventMapper.selectList(new LambdaQueryWrapper<TaskEventEntity>()
                .eq(TaskEventEntity::getTraceId, traceId))
                .stream().map(this::toDomain).toList();
    }

    private TaskEvent toDomain(TaskEventEntity entity) {
        TaskStatus fromStatus = entity.getFromStatus() == null ? null : TaskEnumMapper.statusFromCode(entity.getFromStatus());
        TaskStatus toStatus = entity.getToStatus() == null ? null : TaskEnumMapper.statusFromCode(entity.getToStatus());
        return new TaskEvent(entity.getId(), entity.getTaskId(), entity.getAttemptNo(),
                TaskEnumMapper.eventTypeFromCode(entity.getEventType()), fromStatus, toStatus,
                entity.getStage(), entity.getProgress(), entity.getMessage(), parsePayload(entity.getPayload()),
                TaskOperatorType.valueOf(entity.getOperatorType()), entity.getOperatorId(), entity.getTraceId(),
                entity.getOccurredAt(), entity.getCreatedAt());
    }

    private TaskEventEntity toEntity(TaskEvent event) {
        TaskEventEntity entity = new TaskEventEntity();
        entity.setId(event.getId());
        entity.setTaskId(event.getTaskId());
        entity.setAttemptNo(event.getAttemptNo());
        entity.setEventType(event.getEventType().getCode());
        entity.setFromStatus(event.getFromStatus() == null ? null : event.getFromStatus().getCode());
        entity.setToStatus(event.getToStatus() == null ? null : event.getToStatus().getCode());
        entity.setStage(event.getStage());
        entity.setProgress(event.getProgress());
        entity.setMessage(event.getMessage());
        entity.setPayload(writePayload(event.getPayload()));
        entity.setOperatorType(event.getOperatorType().name());
        entity.setOperatorId(event.getOperatorId());
        entity.setTraceId(event.getTraceId());
        entity.setOccurredAt(event.getOccurredAt());
        entity.setCreatedAt(event.getCreatedAt());
        return entity;
    }

    private Map<String, Object> parsePayload(String payload) {
        if (payload == null) {
            return null;
        }
        try {
            return objectMapper.readValue(payload, PAYLOAD_TYPE);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("任务事件 payload 解析失败", ex);
        }
    }

    private String writePayload(Map<String, Object> payload) {
        if (payload == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("任务事件 payload 序列化失败", ex);
        }
    }
}
