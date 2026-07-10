package com.aiinterview.platform.infrastructure.persistence.task;

import com.aiinterview.platform.infrastructure.persistence.task.entity.TaskAttemptEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 任务执行尝试 MyBatis-Plus Mapper。
 */
@Mapper
public interface TaskAttemptMapper extends BaseMapper<TaskAttemptEntity> {
}
