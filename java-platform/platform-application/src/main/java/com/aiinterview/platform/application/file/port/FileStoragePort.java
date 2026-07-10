package com.aiinterview.platform.application.file.port;

import java.io.InputStream;

/**
 * 文件对象存储端口。
 */
public interface FileStoragePort {

    void store(String objectKey, InputStream content, long size);

    InputStream retrieve(String objectKey);
}
