package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskEventResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 任务事件响应。
 */
public record TaskEventResponse(
        @JsonProperty("event_type") String eventType,
        String message,
        @JsonProperty("occurred_at") Instant occurredAt
) {
    static TaskEventResponse from(TaskEventResult result) {
        return new TaskEventResponse(result.eventType(), result.message(), result.occurredAt());
    }
}
