package com.aiinterview.platform.domain.common;

import java.time.Instant;

/**
 * 可审计对象接口。
 * 核心实体需记录创建时间和更新时间。
 */
public interface Auditable {
    Instant getCreatedAt();
    Instant getUpdatedAt();
}
