package com.aiinterview.platform.domain.asyncmessage;

import java.util.Arrays;
import java.util.Optional;

/**
 * 死信来源类型。
 */
public enum DeadLetterSourceType {
    OUTBOX("outbox"),
    INBOX("inbox"),
    WORKER_FAILURE("worker_failure"),
    STALE_PROCESSING("stale_processing");

    private final String code;

    DeadLetterSourceType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }

    public static Optional<DeadLetterSourceType> fromCode(String code) {
        return Arrays.stream(values())
                .filter(type -> type.code.equals(code))
                .findFirst();
    }
}
