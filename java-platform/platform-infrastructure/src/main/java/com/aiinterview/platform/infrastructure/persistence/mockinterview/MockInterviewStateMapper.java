package com.aiinterview.platform.infrastructure.persistence.mockinterview;

import com.aiinterview.platform.infrastructure.persistence.mockinterview.entity.MockInterviewStateEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 模拟面试中间状态 MyBatis-Plus Mapper。
 */
@Mapper
public interface MockInterviewStateMapper extends BaseMapper<MockInterviewStateEntity> {
}
