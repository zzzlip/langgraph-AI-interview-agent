package com.aiinterview.platform.domain.common;

/**
 * 错误码接口。
 * 所有错误码枚举必须实现此接口，确保错误码集中管理。
 */
public interface ErrorCode {
    int getCode();
    String getMessage();
}
