package com.aiinterview.platform.domain.file;

/**
 * 文件归属校验失败。
 */
public class FileOwnershipException extends RuntimeException {

    public FileOwnershipException(String message) {
        super(message);
    }
}
