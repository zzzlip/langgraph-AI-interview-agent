package com.aiinterview.platform.infrastructure.persistence.file;

import com.aiinterview.platform.domain.file.FileBindType;
import com.aiinterview.platform.domain.file.FileBinding;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.task.PageResult;
import com.aiinterview.platform.infrastructure.persistence.file.entity.FileBindingEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * 文件绑定仓储基础设施实现。
 */
@Component
public class FileBindingRepositoryImpl implements FileBindingRepository {

    private final FileBindingMapper bindingMapper;

    public FileBindingRepositoryImpl(FileBindingMapper bindingMapper) {
        this.bindingMapper = bindingMapper;
    }

    @Override
    public void save(FileBinding binding) {
        bindingMapper.insert(toEntity(binding));
    }

    @Override
    public Optional<FileBinding> findByUniqueKey(FileBinding.FileBindingUniqueKey uniqueKey) {
        return bindingMapper.selectList(new LambdaQueryWrapper<FileBindingEntity>()
                        .eq(FileBindingEntity::getFileId, uniqueKey.fileId())
                        .eq(FileBindingEntity::getTaskId, uniqueKey.taskId())
                        .eq(FileBindingEntity::getBindType, uniqueKey.bindType().getCode())
                        .eq(FileBindingEntity::getAttemptNo, uniqueKey.attemptNo()))
                .stream()
                .findFirst()
                .map(this::toDomain);
    }

    @Override
    public List<FileBinding> findByTaskId(String taskId) {
        return bindingMapper.selectList(new LambdaQueryWrapper<FileBindingEntity>()
                        .eq(FileBindingEntity::getTaskId, taskId)
                        .orderByAsc(FileBindingEntity::getCreatedAt))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public List<FileBinding> findByTaskIdAndBindType(String taskId, FileBindType bindType) {
        return bindingMapper.selectList(new LambdaQueryWrapper<FileBindingEntity>()
                        .eq(FileBindingEntity::getTaskId, taskId)
                        .eq(FileBindingEntity::getBindType, bindType.getCode())
                        .orderByAsc(FileBindingEntity::getCreatedAt))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public PageResult<FileBinding> findByTaskId(String taskId, int page, int pageSize) {
        LambdaQueryWrapper<FileBindingEntity> wrapper = new LambdaQueryWrapper<FileBindingEntity>()
                .eq(FileBindingEntity::getTaskId, taskId)
                .orderByAsc(FileBindingEntity::getCreatedAt);
        return pageResult(wrapper, page, pageSize);
    }

    @Override
    public PageResult<FileBinding> findByTaskIdAndBindType(String taskId, FileBindType bindType, int page, int pageSize) {
        LambdaQueryWrapper<FileBindingEntity> wrapper = new LambdaQueryWrapper<FileBindingEntity>()
                .eq(FileBindingEntity::getTaskId, taskId)
                .eq(FileBindingEntity::getBindType, bindType.getCode())
                .orderByAsc(FileBindingEntity::getCreatedAt);
        return pageResult(wrapper, page, pageSize);
    }

    private PageResult<FileBinding> pageResult(LambdaQueryWrapper<FileBindingEntity> wrapper, int page, int pageSize) {
        long total = bindingMapper.selectCount(wrapper);
        int offset = Math.max(page - 1, 0) * pageSize;
        List<FileBinding> items = bindingMapper.selectList(wrapper.last("LIMIT " + pageSize + " OFFSET " + offset))
                .stream()
                .map(this::toDomain)
                .toList();
        return new PageResult<>(items, total, page, pageSize);
    }

    private FileBinding toDomain(FileBindingEntity entity) {
        FileBindType bindType = FileEnumMapper.bindTypeFromCode(entity.getBindType());
        if (bindType == FileBindType.INPUT) {
            return FileBinding.createInput(entity.getId(), entity.getFileId(), entity.getTaskId(), entity.getUserId(), entity.getAttemptNo(), entity.getCreatedAt());
        }
        return FileBinding.createWorkerOutput(entity.getId(), entity.getFileId(), entity.getTaskId(), entity.getUserId(), bindType, entity.getAttemptNo(), entity.getCreatedAt());
    }

    private FileBindingEntity toEntity(FileBinding binding) {
        FileBindingEntity entity = new FileBindingEntity();
        entity.setId(binding.getId());
        entity.setFileId(binding.getFileId());
        entity.setTaskId(binding.getTaskId());
        entity.setUserId(binding.getUserId());
        entity.setBindType(binding.getBindType().getCode());
        entity.setAttemptNo(binding.getAttemptNo());
        entity.setCreatedAt(binding.getCreatedAt());
        return entity;
    }
}
