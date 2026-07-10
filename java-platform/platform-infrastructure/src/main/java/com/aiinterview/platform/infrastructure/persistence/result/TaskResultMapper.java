package com.aiinterview.platform.infrastructure.persistence.result;

import com.aiinterview.platform.infrastructure.persistence.result.entity.TaskResultEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 任务结果 MyBatis-Plus Mapper。
 */
@Mapper
public interface TaskResultMapper extends BaseMapper<TaskResultEntity> {
}
