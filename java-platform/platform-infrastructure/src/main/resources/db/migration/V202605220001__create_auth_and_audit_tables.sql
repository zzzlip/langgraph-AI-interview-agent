-- spec02: 创建账号表与审计事件表
CREATE TABLE `account` (
    `id`            VARCHAR(36)     NOT NULL COMMENT 'UUIDv7 主键',
    `username`      VARCHAR(64)     NOT NULL COMMENT '用户名，全局唯一',
    `password_hash` VARCHAR(256)    NOT NULL COMMENT 'BCrypt 密码哈希',
    `role`          VARCHAR(16)     NOT NULL DEFAULT 'USER' COMMENT '角色：USER / ADMIN',
    `status`        VARCHAR(16)     NOT NULL DEFAULT 'NORMAL' COMMENT '状态：NORMAL / DISABLED',
    `created_at`    DATETIME(3)     NOT NULL COMMENT '创建时间',
    `updated_at`    DATETIME(3)     NOT NULL COMMENT '更新时间',
    `deleted_at`    DATETIME(3)     DEFAULT NULL COMMENT '软删除时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_account_username` (`username`),
    INDEX `idx_account_status` (`status`),
    INDEX `idx_account_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='账号表';

CREATE TABLE `audit_event` (
    `id`            VARCHAR(36)     NOT NULL COMMENT 'UUIDv7 主键',
    `event_type`    VARCHAR(40)     NOT NULL COMMENT '事件类型枚举',
    `actor_id`      VARCHAR(36)     NOT NULL COMMENT '操作人 ID',
    `actor_role`    VARCHAR(16)     DEFAULT NULL COMMENT '操作人角色',
    `target_type`   VARCHAR(40)     DEFAULT NULL COMMENT '目标资源类型：task / file / report / admin / account',
    `target_id`     VARCHAR(128)    DEFAULT NULL COMMENT '目标资源标识',
    `action`        VARCHAR(64)     NOT NULL COMMENT '操作名称',
    `success`       TINYINT(1)      NOT NULL COMMENT '操作是否成功：1 成功 / 0 失败',
    `failure_code`  VARCHAR(40)     DEFAULT NULL COMMENT '失败原因码',
    `trace_id`      VARCHAR(64)     DEFAULT NULL COMMENT '请求链路 ID',
    `source_ip`     VARCHAR(45)     DEFAULT NULL COMMENT '请求来源 IP（支持 IPv6）',
    `user_agent`    VARCHAR(256)    DEFAULT NULL COMMENT '用户代理摘要',
    `occurred_at`   DATETIME(3)     NOT NULL COMMENT '事件发生时间',
    `created_at`    DATETIME(3)     NOT NULL COMMENT '记录创建时间',
    PRIMARY KEY (`id`),
    INDEX `idx_audit_event_type` (`event_type`),
    INDEX `idx_audit_actor_id` (`actor_id`),
    INDEX `idx_audit_target` (`target_type`, `target_id`),
    INDEX `idx_audit_occurred_at` (`occurred_at`),
    INDEX `idx_audit_trace_id` (`trace_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审计事件表';
