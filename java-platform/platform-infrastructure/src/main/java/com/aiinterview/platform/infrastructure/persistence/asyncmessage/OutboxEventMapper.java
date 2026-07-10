package com.aiinterview.platform.infrastructure.persistence.asyncmessage;

import com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity.OutboxEventEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * Outbox 事件 MyBatis-Plus Mapper。
 */
@Mapper
public interface OutboxEventMapper extends BaseMapper<OutboxEventEntity> {
}
