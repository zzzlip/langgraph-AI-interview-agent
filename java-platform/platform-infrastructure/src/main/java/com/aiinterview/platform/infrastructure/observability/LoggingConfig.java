package com.aiinterview.platform.infrastructure.observability;

import org.slf4j.MDC;
import org.springframework.context.annotation.Configuration;

/**
 * 日志配置类。
 *
 * 用于 Logback JSON 编码的 MDC 字段管理。
 * 实际 JSON 编码在 logback-spring.xml 中使用 LogstashEncoder 完成。
 *
 * MDC 字段约定：
 * - trace_id: 请求追踪 ID（由 TraceIdFilter 设置）
 * - task_id: 任务 ID（由应用层设置）
 * - stage: 执行阶段
 * - event_id: 事件 ID
 * - error_code: 错误码
 */
@Configuration
public class LoggingConfig {

    public static final String MDC_TASK_ID = "task_id";
    public static final String MDC_STAGE = "stage";
    public static final String MDC_EVENT_ID = "event_id";
    public static final String MDC_ERROR_CODE = "error_code";

    public static void putTaskId(String taskId) {
        if (taskId != null) {
            MDC.put(MDC_TASK_ID, taskId);
        }
    }

    public static void putStage(String stage) {
        if (stage != null) {
            MDC.put(MDC_STAGE, stage);
        }
    }

    public static void putEventId(String eventId) {
        if (eventId != null) {
            MDC.put(MDC_EVENT_ID, eventId);
        }
    }

    public static void putErrorCode(String errorCode) {
        if (errorCode != null) {
            MDC.put(MDC_ERROR_CODE, errorCode);
        }
    }

    public static void clear() {
        MDC.remove(MDC_TASK_ID);
        MDC.remove(MDC_STAGE);
        MDC.remove(MDC_EVENT_ID);
        MDC.remove(MDC_ERROR_CODE);
    }
}
