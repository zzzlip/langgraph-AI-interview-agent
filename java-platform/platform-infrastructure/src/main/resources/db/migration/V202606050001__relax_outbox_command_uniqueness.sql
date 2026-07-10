ALTER TABLE `outbox_event`
    DROP INDEX `uk_outbox_event_task_attempt_event`;

CREATE INDEX `idx_outbox_event_task_attempt_event`
    ON `outbox_event` (`task_id`, `attempt_no`, `event_type`);
