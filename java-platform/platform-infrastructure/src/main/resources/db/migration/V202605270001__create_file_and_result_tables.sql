-- spec04: 创建文件、文件任务绑定和任务结果表
CREATE TABLE `file` (
    `id`               VARCHAR(36)  NOT NULL COMMENT 'UUIDv7 文件 ID',
    `user_id`          VARCHAR(36)  NOT NULL COMMENT '文件归属用户',
    `original_name`    VARCHAR(255) NOT NULL COMMENT '原始文件名',
    `stored_filename`  VARCHAR(255) NOT NULL COMMENT '存储文件名',
    `file_type`        VARCHAR(32)  NOT NULL COMMENT '文件类型',
    `file_purpose`     VARCHAR(32)  NOT NULL COMMENT '文件用途',
    `file_size`        BIGINT       NOT NULL COMMENT '文件大小',
    `mime_type`        VARCHAR(128) NOT NULL COMMENT 'MIME 类型',
    `source_device_id` VARCHAR(128) DEFAULT NULL COMMENT '来源设备标识',
    `file_status`      VARCHAR(32)  NOT NULL DEFAULT 'uploading' COMMENT '文件状态',
    `object_key`       VARCHAR(512) NOT NULL COMMENT '对象存储 key',
    `checksum`         VARCHAR(128) DEFAULT NULL COMMENT '文件校验值',
    `failure_reason`   VARCHAR(500) DEFAULT NULL COMMENT '脱敏失败原因',
    `created_at`       DATETIME(3)  NOT NULL COMMENT '创建时间',
    `updated_at`       DATETIME(3)  NOT NULL COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_file_object_key` (`object_key`),
    INDEX `idx_file_user_created` (`user_id`, `created_at`),
    INDEX `idx_file_type_purpose` (`file_type`, `file_purpose`),
    CONSTRAINT `ck_file_source_device_id_safe` CHECK (`source_device_id` IS NULL OR `source_device_id` REGEXP '^[A-Za-z0-9._:-]{1,128}$')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='文件元数据表';

CREATE TABLE `file_task_binding` (
    `id`         VARCHAR(36) NOT NULL COMMENT 'UUIDv7 绑定 ID',
    `file_id`    VARCHAR(36) NOT NULL COMMENT '文件 ID',
    `task_id`    VARCHAR(36) NOT NULL COMMENT '任务 ID',
    `user_id`    VARCHAR(36) NOT NULL COMMENT '用户 ID',
    `bind_type`  VARCHAR(32) NOT NULL COMMENT '绑定类型',
    `attempt_no` INT         NOT NULL DEFAULT 0 COMMENT '任务尝试序号',
    `created_at` DATETIME(3) NOT NULL COMMENT '绑定时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_file_task_bind` (`file_id`, `task_id`, `bind_type`, `attempt_no`),
    INDEX `idx_binding_task_type` (`task_id`, `bind_type`),
    INDEX `idx_binding_user_file` (`user_id`, `file_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='文件任务绑定表';

CREATE TABLE `task_result` (
    `id`            VARCHAR(36) NOT NULL COMMENT 'UUIDv7 结果 ID',
    `task_id`       VARCHAR(36) NOT NULL COMMENT '任务 ID',
    `attempt_no`    INT         NOT NULL COMMENT '任务尝试序号',
    `result_type`   VARCHAR(32) NOT NULL COMMENT '结果类型',
    `summary`       JSON        NOT NULL COMMENT '结果摘要 JSON',
    `result_status` VARCHAR(32) NOT NULL DEFAULT 'completed' COMMENT '结果状态',
    `created_at`    DATETIME(3) NOT NULL COMMENT '创建时间',
    `updated_at`    DATETIME(3) NOT NULL COMMENT '更新时间',
    `completed_at`  DATETIME(3) DEFAULT NULL COMMENT '完成时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_result_task_attempt` (`task_id`, `attempt_no`),
    INDEX `idx_result_type_created` (`result_type`, `created_at`),
    CONSTRAINT `ck_task_result_status_completed` CHECK (`result_status` = 'completed'),
    CONSTRAINT `ck_task_result_summary_schema` CHECK (JSON_EXTRACT(`summary`, '$.schema_version') IS NOT NULL)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务结构化结果表';
