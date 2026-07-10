package com.aiinterview.platform.application.asyncmessage.dto;

/**
 * 死信人工处理结果。
 */
public record DeadLetterHandledResult(String deadLetterId, boolean manualHandled) {
}
