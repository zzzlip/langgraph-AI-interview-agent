package com.aiinterview.platform.infrastructure.persistence.file;

import com.aiinterview.platform.infrastructure.persistence.file.entity.FileEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 文件 MyBatis-Plus Mapper。
 */
@Mapper
public interface FileMapper extends BaseMapper<FileEntity> {
}
