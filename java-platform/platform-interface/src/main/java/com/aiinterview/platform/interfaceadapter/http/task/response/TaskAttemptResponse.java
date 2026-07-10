package com.aiinterview.platform.interfaceadapter.http.task.response;

import com.aiinterview.platform.application.task.dto.TaskAttemptResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;

/**
 * 任务执行尝试响应。
 */
public record TaskAttemptResponse(
        @JsonProperty("attempt_no") int attemptNo,
        String status,
        @JsonProperty("started_at") Instant startedAt,
        @JsonProperty("finished_at") Instant finishedAt,
        @JsonProperty("failed_at") Instant failedAt
) {
    static TaskAttemptResponse from(TaskAttemptResult result) {
        return new TaskAttemptResponse(
                result.attemptNo(),
                result.status(),
                result.startedAt(),
                result.finishedAt(),
                result.failedAt()
        );
    }
}
