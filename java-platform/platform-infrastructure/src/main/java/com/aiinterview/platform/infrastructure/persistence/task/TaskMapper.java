package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 任务 MyBatis-Plus Mapper。
 */
@Mapper
public interface TaskMapper extends BaseMapper<TaskEntity> {
}
