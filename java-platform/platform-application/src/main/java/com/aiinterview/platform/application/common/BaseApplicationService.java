package com.aiinterview.platform.application.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 应用服务基类。
 *
 * 提供统一的日志访问和事务辅助能力。
 * 应用服务继承此类，专注于用例编排。
 */
public abstract class BaseApplicationService {

    protected final Logger log = LoggerFactory.getLogger(getClass());

    /**
     * 获取当前 trace ID（由 TraceIdFilter 在基础设施层注入 MDC）。
     */
    protected String currentTraceId() {
        return java.util.Optional.ofNullable(
                org.slf4j.MDC.get("trace_id")
        ).orElse("");
    }

    /**
     * 获取当前请求来源 IP（由基础设施层注入 MDC）。
     */
    protected String currentSourceIp() {
        return java.util.Optional.ofNullable(
                org.slf4j.MDC.get("source_ip")
        ).orElse("");
    }
}
