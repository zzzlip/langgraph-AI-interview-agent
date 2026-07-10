package com.aiinterview.platform.domain.task;

/**
 * 任务幂等唯一约束冲突。
 */
public class DuplicateTaskIdempotencyException extends RuntimeException {

    public DuplicateTaskIdempotencyException(String message) {
        super(message);
    }
}
