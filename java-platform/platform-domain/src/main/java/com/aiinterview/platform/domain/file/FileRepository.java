package com.aiinterview.platform.domain.file;

import java.util.List;
import java.util.Optional;

/**
 * 文件聚合仓储接口。
 */
public interface FileRepository {

    void save(File file);

    Optional<File> findById(String fileId);

    List<File> findByIds(List<String> fileIds);

    List<File> findByUserId(String userId);

    boolean updateStatus(File file);
}
