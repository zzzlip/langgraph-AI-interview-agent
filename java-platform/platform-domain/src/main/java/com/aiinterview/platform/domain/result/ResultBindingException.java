package com.aiinterview.platform.domain.result;

/**
 * 结果绑定或摘要校验失败。
 */
public class ResultBindingException extends RuntimeException {

    public ResultBindingException(String message) {
        super(message);
    }
}
