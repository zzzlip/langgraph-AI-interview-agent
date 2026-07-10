package com.aiinterview.platform.infrastructure.persistence.auth;

import com.aiinterview.platform.infrastructure.persistence.auth.entity.AccountEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 账号 MyBatis-Plus Mapper。
 */
@Mapper
public interface AccountMapper extends BaseMapper<AccountEntity> {
}
