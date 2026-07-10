package com.aiinterview.platform.domain.asyncmessage;

import java.util.Arrays;
import java.util.Optional;

/**
 * 异步消息投递状态。
 */
public enum AsyncDeliveryStatus {
    PENDING("pending", false),
    DELIVERING("delivering", false),
    DELIVERED("delivered", true),
    FAILED("failed", false),
    DEAD("dead", true);

    private final String code;
    private final boolean finalStatus;

    AsyncDeliveryStatus(String code, boolean finalStatus) {
        this.code = code;
        this.finalStatus = finalStatus;
    }

    public String getCode() {
        return code;
    }

    public boolean isFinal() {
        return finalStatus;
    }

    public static Optional<AsyncDeliveryStatus> fromCode(String code) {
        return Arrays.stream(values())
                .filter(status -> status.code.equals(code))
                .findFirst();
    }
}
