package com.aiinterview.platform.infrastructure.persistence.file;

import com.aiinterview.platform.application.file.dto.AdminFileStatusItem;
import com.aiinterview.platform.application.file.dto.AdminFileStatusPageResult;
import com.aiinterview.platform.application.file.port.AdminFileStatusQuery;
import com.aiinterview.platform.application.file.port.AdminFileStatusQueryPort;
import com.aiinterview.platform.infrastructure.persistence.file.entity.FileBindingEntity;
import com.aiinterview.platform.infrastructure.persistence.file.entity.FileEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.List;
import org.springframework.stereotype.Component;

/**
 * 管理员文件状态查询基础设施适配器。
 */
@Component
public class AdminFileStatusQueryAdapter implements AdminFileStatusQueryPort {

    private final FileMapper fileMapper;
    private final FileBindingMapper bindingMapper;

    public AdminFileStatusQueryAdapter(FileMapper fileMapper, FileBindingMapper bindingMapper) {
        this.fileMapper = fileMapper;
        this.bindingMapper = bindingMapper;
    }

    @Override
    public AdminFileStatusPageResult search(AdminFileStatusQuery query) {
        List<String> fileIds = fileIdsByTask(query.taskId());
        if (query.taskId() != null && fileIds.isEmpty()) {
            return new AdminFileStatusPageResult(List.of(), 0, query.page(), query.pageSize());
        }
        LambdaQueryWrapper<FileEntity> countWrapper = wrapper(query, fileIds);
        long total = fileMapper.selectCount(countWrapper);
        List<AdminFileStatusItem> items = fileMapper.selectList(wrapper(query, fileIds)
                        .orderByDesc(FileEntity::getCreatedAt)
                        .orderByDesc(FileEntity::getId)
                        .last("LIMIT " + query.pageSize() + " OFFSET " + offset(query)))
                .stream()
                .map(file -> toItem(file, query.taskId()))
                .toList();
        return new AdminFileStatusPageResult(items, total, query.page(), query.pageSize());
    }

    private LambdaQueryWrapper<FileEntity> wrapper(AdminFileStatusQuery query, List<String> fileIds) {
        LambdaQueryWrapper<FileEntity> wrapper = new LambdaQueryWrapper<>();
        if (query.fileStatus() != null) {
            wrapper.eq(FileEntity::getFileStatus, query.fileStatus());
        }
        if (query.filePurpose() != null) {
            wrapper.eq(FileEntity::getFilePurpose, query.filePurpose());
        }
        if (query.taskId() != null) {
            wrapper.in(FileEntity::getId, fileIds);
        }
        return wrapper;
    }

    private List<String> fileIdsByTask(String taskId) {
        if (taskId == null) {
            return List.of();
        }
        return bindingMapper.selectList(new LambdaQueryWrapper<FileBindingEntity>()
                        .eq(FileBindingEntity::getTaskId, taskId))
                .stream()
                .map(FileBindingEntity::getFileId)
                .distinct()
                .toList();
    }

    private AdminFileStatusItem toItem(FileEntity file, String filteredTaskId) {
        String taskId = filteredTaskId == null ? firstTaskId(file.getId()) : filteredTaskId;
        return new AdminFileStatusItem(
                file.getId(),
                taskId,
                file.getFileType(),
                file.getFilePurpose(),
                file.getFileStatus(),
                file.getFailureReason(),
                file.getCreatedAt(),
                file.getUpdatedAt()
        );
    }

    private String firstTaskId(String fileId) {
        return bindingMapper.selectList(new LambdaQueryWrapper<FileBindingEntity>()
                        .eq(FileBindingEntity::getFileId, fileId)
                        .orderByAsc(FileBindingEntity::getCreatedAt)
                        .last("LIMIT 1"))
                .stream()
                .findFirst()
                .map(FileBindingEntity::getTaskId)
                .orElse(null);
    }

    private static int offset(AdminFileStatusQuery query) {
        return (query.page() - 1) * query.pageSize();
    }
}
