package com.aiinterview.platform.domain.common;

import java.util.Objects;

/**
 * 通用返回结果。
 * 用于统一后端响应结构，包含业务状态码、消息和数据。
 */
public class BaseResult<T> {

    private int code;
    private String message;
    private T data;

    public BaseResult() {
    }

    public BaseResult(int code, String message, T data) {
        this.code = code;
        this.message = message;
        this.data = data;
    }

    /** 创建成功结果 */
    public static <T> BaseResult<T> success() {
        return new BaseResult<>(0, "success", null);
    }

    /** 创建成功结果并携带数据 */
    public static <T> BaseResult<T> success(T data) {
        return new BaseResult<>(0, "success", data);
    }

    /** 创建失败结果 */
    public static <T> BaseResult<T> error(int code, String message) {
        return new BaseResult<>(code, message, null);
    }

    /** 创建失败结果，携带错误码枚举 */
    public static <T> BaseResult<T> error(ErrorCode errorCode) {
        return new BaseResult<>(errorCode.getCode(), errorCode.getMessage(), null);
    }

    /** 创建失败结果，携带自定义消息覆盖错误码默认消息 */
    public static <T> BaseResult<T> error(ErrorCode errorCode, String message) {
        return new BaseResult<>(errorCode.getCode(), message, null);
    }

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BaseResult<?> that = (BaseResult<?>) o;
        return code == that.code && Objects.equals(message, that.message) && Objects.equals(data, that.data);
    }

    @Override
    public int hashCode() {
        return Objects.hash(code, message, data);
    }

    @Override
    public String toString() {
        return "BaseResult{" +
                "code=" + code +
                ", message='" + message + '\'' +
                ", data=" + data +
                '}';
    }
}
