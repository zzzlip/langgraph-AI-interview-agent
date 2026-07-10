package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskEventEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 任务事件 MyBatis-Plus Mapper。
 */
@Mapper
public interface TaskEventMapper extends BaseMapper<TaskEventEntity> {
}
