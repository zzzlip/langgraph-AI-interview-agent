package com.aiinterview.platform.application.task.port;

/**
 * 业务资源归属校验端口。
 */
public interface ResourceOwnershipPort {

    boolean isOwnedByUser(String userId, String resourceType, String resourceId);
}
