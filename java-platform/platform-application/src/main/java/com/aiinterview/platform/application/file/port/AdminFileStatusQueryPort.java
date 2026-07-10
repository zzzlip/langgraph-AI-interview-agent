package com.aiinterview.platform.application.file.port;

import com.aiinterview.platform.application.file.dto.AdminFileStatusPageResult;

/**
 * 管理员文件状态查询端口。
 */
public interface AdminFileStatusQueryPort {

    AdminFileStatusPageResult search(AdminFileStatusQuery query);
}
