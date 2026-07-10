package com.aiinterview.platform.infrastructure.messaging.rabbitmq;

import com.aiinterview.platform.application.asyncmessage.AsyncMessageException;
import com.aiinterview.platform.application.asyncmessage.command.WorkerFeedbackEventCommand;
import com.aiinterview.platform.infrastructure.asyncmessage.AlgorithmProblemSetFeedbackDispatcher;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

/**
 * Worker 反馈 RabbitMQ 消费入口。
 */
@Component
public class RabbitWorkerFeedbackListener {

    private final AlgorithmProblemSetFeedbackDispatcher dispatcher;
    private final ObjectMapper objectMapper;

    public RabbitWorkerFeedbackListener(AlgorithmProblemSetFeedbackDispatcher dispatcher, ObjectMapper objectMapper) {
        this.dispatcher = dispatcher;
        this.objectMapper = objectMapper;
    }

    @RabbitListener(queues = RabbitMqTopologyConfig.PLATFORM_TASK_FEEDBACK_QUEUE)
    public void onMessage(String body) {
        try {
            JsonNode root = objectMapper.readTree(body);
            dispatcher.dispatch(new WorkerFeedbackEventCommand(
                    intValue(root, "schema_version"),
                    text(root, "event_id"),
                    text(root, "event_type"),
                    text(root, "task_id"),
                    text(root, "task_type"),
                    intValue(root, "attempt"),
                    text(root, "trace_id"),
                    body,
                    payloadSummary(root)
            ));
        } catch (AsyncMessageException ex) {
            throw ex;
        } catch (Exception ex) {
            throw new AsyncMessageException(74005, "Worker 事件格式非法");
        }
    }

    private String payloadSummary(JsonNode root) {
        JsonNode payload = root.path("payload");
        ObjectNode summary = objectMapper.createObjectNode();
        copyText(payload, summary, "worker_id");
        copyText(payload, summary, "command_event_id");
        copyText(payload, summary, "consumed_at");
        copyText(payload, summary, "status");
        copyText(payload, summary, "stage");
        copyInt(payload, summary, "progress");
        copyText(payload, summary, "message");
        copyText(payload, summary, "failure_code");
        String failedStage = textValue(payload, "failed_stage");
        if (failedStage != null) {
            summary.put("failed_stage", failedStage);
            summary.put("stage", failedStage);
        }
        return summary.toString();
    }

    private static void copyText(JsonNode source, ObjectNode target, String field) {
        String value = textValue(source, field);
        if (value != null) {
            target.put(field, value);
        }
    }

    private static void copyInt(JsonNode source, ObjectNode target, String field) {
        JsonNode value = source.path(field);
        if (value.isInt()) {
            target.put(field, value.asInt());
        }
    }

    private static String text(JsonNode root, String field) {
        JsonNode value = root.path(field);
        return value.isTextual() ? value.asText() : "";
    }

    private static String textValue(JsonNode root, String field) {
        JsonNode value = root.path(field);
        return value.isTextual() && !value.asText().isBlank() ? value.asText() : null;
    }

    private static int intValue(JsonNode root, String field) {
        JsonNode value = root.path(field);
        return value.isInt() ? value.asInt() : 0;
    }
}
