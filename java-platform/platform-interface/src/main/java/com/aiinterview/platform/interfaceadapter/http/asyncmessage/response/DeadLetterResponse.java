package com.aiinterview.platform.interfaceadapter.http.asyncmessage.response;

import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterHandledResult;
import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterItemResult;
import com.aiinterview.platform.application.asyncmessage.dto.DeadLetterPageResult;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.List;

/**
 * 管理员死信响应。
 */
public final class DeadLetterResponse {

    private DeadLetterResponse() {
    }

    public record Page(List<Item> items, long total, int page, @JsonProperty("page_size") int pageSize) {
        public static Page from(DeadLetterPageResult result) {
            return new Page(result.items().stream().map(Item::from).toList(), result.total(), result.page(),
                    result.pageSize());
        }
    }

    public record Item(
            @JsonProperty("dead_letter_id") String deadLetterId,
            @JsonProperty("source_type") String sourceType,
            @JsonProperty("task_id") String taskId,
            @JsonProperty("task_type") String taskType,
            int attempt,
            @JsonProperty("failure_stage") String failureStage,
            @JsonProperty("failure_reason") String failureReason,
            boolean retryable,
            @JsonProperty("manual_handled") boolean manualHandled,
            @JsonProperty("created_at") Instant createdAt
    ) {
        static Item from(DeadLetterItemResult item) {
            return new Item(item.deadLetterId(), item.sourceType(), item.taskId(), item.taskType(), item.attempt(),
                    item.failureStage(), item.failureReason(), item.retryable(), item.manualHandled(),
                    item.createdAt());
        }
    }

    public record Handled(@JsonProperty("dead_letter_id") String deadLetterId,
                          @JsonProperty("manual_handled") boolean manualHandled) {
        public static Handled from(DeadLetterHandledResult result) {
            return new Handled(result.deadLetterId(), result.manualHandled());
        }
    }
}
