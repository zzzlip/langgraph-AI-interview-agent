package com.aiinterview.platform.domain.asyncmessage;

import java.util.Arrays;
import java.util.Optional;

/**
 * 异步消息消费状态。
 */
public enum AsyncConsumptionStatus {
    RECEIVED("received", false),
    PROCESSING("processing", false),
    PROCESSED("processed", true),
    IGNORED("ignored", true),
    FAILED("failed", false),
    DEAD("dead", true);

    private final String code;
    private final boolean finalStatus;

    AsyncConsumptionStatus(String code, boolean finalStatus) {
        this.code = code;
        this.finalStatus = finalStatus;
    }

    public String getCode() {
        return code;
    }

    public boolean isFinal() {
        return finalStatus;
    }

    public static Optional<AsyncConsumptionStatus> fromCode(String code) {
        return Arrays.stream(values())
                .filter(status -> status.code.equals(code))
                .findFirst();
    }
}
