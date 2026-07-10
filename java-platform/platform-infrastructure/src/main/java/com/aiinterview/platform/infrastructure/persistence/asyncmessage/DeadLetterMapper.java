package com.aiinterview.platform.infrastructure.persistence.asyncmessage;

import com.aiinterview.platform.infrastructure.persistence.asyncmessage.entity.DeadLetterEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 死信记录 MyBatis-Plus Mapper。
 */
@Mapper
public interface DeadLetterMapper extends BaseMapper<DeadLetterEntity> {
}
