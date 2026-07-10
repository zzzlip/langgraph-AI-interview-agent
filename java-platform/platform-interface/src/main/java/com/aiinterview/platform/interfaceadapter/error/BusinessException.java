package com.aiinterview.platform.interfaceadapter.error;

import com.aiinterview.platform.domain.common.ErrorCode;

/**
 * 业务异常基类。
 *
 * 所有业务异常继承此类，统一异常处理时根据异常类型
 * 匹配对应的 HTTP 状态码和错误码。
 */
public class BusinessException extends RuntimeException {

    private final int code;

    public BusinessException(ErrorCode errorCode) {
        super(errorCode.getMessage());
        this.code = errorCode.getCode();
    }

    public BusinessException(ErrorCode errorCode, String message) {
        super(message);
        this.code = errorCode.getCode();
    }

    public BusinessException(int code, String message) {
        super(message);
        this.code = code;
    }

    public int getCode() {
        return code;
    }
}
