package com.aiinterview.platform.domain.file;

import com.aiinterview.platform.domain.task.PageResult;
import java.util.List;
import java.util.Optional;

/**
 * 文件任务绑定仓储接口。
 */
public interface FileBindingRepository {

    void save(FileBinding binding);

    Optional<FileBinding> findByUniqueKey(FileBinding.FileBindingUniqueKey uniqueKey);

    List<FileBinding> findByTaskId(String taskId);

    List<FileBinding> findByTaskIdAndBindType(String taskId, FileBindType bindType);

    PageResult<FileBinding> findByTaskId(String taskId, int page, int pageSize);

    PageResult<FileBinding> findByTaskIdAndBindType(String taskId, FileBindType bindType, int page, int pageSize);
}
