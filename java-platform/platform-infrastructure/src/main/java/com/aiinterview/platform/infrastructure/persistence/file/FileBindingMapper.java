package com.aiinterview.platform.infrastructure.persistence.file;

import com.aiinterview.platform.infrastructure.persistence.file.entity.FileBindingEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 文件绑定 MyBatis-Plus Mapper。
 */
@Mapper
public interface FileBindingMapper extends BaseMapper<FileBindingEntity> {
}
