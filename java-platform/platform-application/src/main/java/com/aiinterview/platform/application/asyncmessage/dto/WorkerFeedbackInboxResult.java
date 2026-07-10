package com.aiinterview.platform.application.asyncmessage.dto;

import com.aiinterview.platform.domain.asyncmessage.InboxEvent;

/**
 * Worker 反馈 Inbox 幂等处理结果。
 */
public record WorkerFeedbackInboxResult(Status status, InboxEvent event) {

    public enum Status {
        RECEIVED,
        DUPLICATE_ACK
    }

    public static WorkerFeedbackInboxResult received(InboxEvent event) {
        return new WorkerFeedbackInboxResult(Status.RECEIVED, event);
    }

    public static WorkerFeedbackInboxResult duplicateAck(InboxEvent event) {
        return new WorkerFeedbackInboxResult(Status.DUPLICATE_ACK, event);
    }
}
