package com.aiinterview.platform.infrastructure.persistence.asyncmessage;

import com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity.InboxEventEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * Inbox 事件 MyBatis-Plus Mapper。
 */
@Mapper
public interface InboxEventMapper extends BaseMapper<InboxEventEntity> {
}
