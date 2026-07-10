package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskIdempotencyEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 任务幂等记录 MyBatis-Plus Mapper。
 */
@Mapper
public interface TaskIdempotencyMapper extends BaseMapper<TaskIdempotencyEntity> {
}
