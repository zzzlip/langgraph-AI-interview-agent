package com.aiinterview.platform.interfaceadapter.error;

/**
 * 校验异常。
 * 对应 HTTP 400 + 错误码 10001（参数校验失败）。
 */
public class ValidationException extends BusinessException {

    private static final int DEFAULT_CODE = 10001;
    private static final String DEFAULT_MESSAGE = "请求参数校验失败";

    public ValidationException() {
        super(DEFAULT_CODE, DEFAULT_MESSAGE);
    }

    public ValidationException(String message) {
        super(DEFAULT_CODE, message);
    }

    public ValidationException(String field, String reason) {
        super(DEFAULT_CODE, "字段 '" + field + "' 校验失败: " + reason);
    }
}
