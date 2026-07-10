package com.aiinterview.platform.infrastructure.persistence.audit;

import com.aiinterview.platform.infrastructure.persistence.audit.entity.AuditEventEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 审计事件 MyBatis-Plus Mapper。
 */
@Mapper
public interface AuditEventMapper extends BaseMapper<AuditEventEntity> {
}
