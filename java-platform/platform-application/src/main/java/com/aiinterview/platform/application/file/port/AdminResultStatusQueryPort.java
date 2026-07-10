package com.aiinterview.platform.application.file.port;

import com.aiinterview.platform.application.file.dto.AdminResultStatusPageResult;

/**
 * 管理员结果状态查询端口。
 */
public interface AdminResultStatusQueryPort {

    AdminResultStatusPageResult search(AdminResultStatusQuery query);
}
