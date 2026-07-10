package com.aiinterview.platform.infrastructure.persistence.file;

import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.file.FileStatus;
import com.aiinterview.platform.infrastructure.persistence.file.entity.FileEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Component;

/**
 * 文件仓储基础设施实现。
 */
@Component
public class FileRepositoryImpl implements FileRepository {

    private final FileMapper fileMapper;

    public FileRepositoryImpl(FileMapper fileMapper) {
        this.fileMapper = fileMapper;
    }

    @Override
    public void save(File file) {
        fileMapper.insert(toEntity(file));
    }

    @Override
    public Optional<File> findById(String fileId) {
        return Optional.ofNullable(fileMapper.selectById(fileId)).map(this::toDomain);
    }

    @Override
    public List<File> findByIds(List<String> fileIds) {
        if (fileIds == null || fileIds.isEmpty()) {
            return List.of();
        }
        return fileMapper.selectBatchIds(fileIds)
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public List<File> findByUserId(String userId) {
        return fileMapper.selectList(new LambdaQueryWrapper<FileEntity>()
                        .eq(FileEntity::getUserId, userId)
                        .orderByDesc(FileEntity::getCreatedAt))
                .stream()
                .map(this::toDomain)
                .toList();
    }

    @Override
    public boolean updateStatus(File file) {
        FileEntity entity = toEntity(file);
        int rows = fileMapper.update(entity, new LambdaUpdateWrapper<FileEntity>()
                .eq(FileEntity::getId, file.getId()));
        return rows > 0;
    }

    private File toDomain(FileEntity entity) {
        File file = File.createUploading(
                entity.getId(),
                entity.getUserId(),
                entity.getOriginalName(),
                entity.getStoredFilename(),
                FileEnumMapper.typeFromCode(entity.getFileType()),
                FileEnumMapper.purposeFromCode(entity.getFilePurpose()),
                entity.getFileSize(),
                entity.getMimeType(),
                entity.getSourceDeviceId(),
                entity.getObjectKey(),
                entity.getCreatedAt()
        );
        if (FileEnumMapper.statusFromCode(entity.getFileStatus()) == FileStatus.AVAILABLE) {
            file.markAvailable(entity.getChecksum(), entity.getUpdatedAt());
        } else if (FileEnumMapper.statusFromCode(entity.getFileStatus()) == FileStatus.FAILED) {
            file.markFailed(entity.getFailureReason(), entity.getUpdatedAt());
        }
        return file;
    }

    private FileEntity toEntity(File file) {
        FileEntity entity = new FileEntity();
        entity.setId(file.getId());
        entity.setUserId(file.getUserId());
        entity.setOriginalName(file.getOriginalName());
        entity.setStoredFilename(file.getStoredFilename());
        entity.setFileType(file.getFileType().getCode());
        entity.setFilePurpose(file.getFilePurpose().getCode());
        entity.setFileSize(file.getFileSize());
        entity.setMimeType(file.getMimeType());
        entity.setSourceDeviceId(file.getSourceDeviceId());
        entity.setFileStatus(file.getFileStatus().getCode());
        entity.setObjectKey(file.getObjectKey());
        entity.setChecksum(file.getChecksum());
        entity.setFailureReason(file.getFailureReason());
        entity.setCreatedAt(file.getCreatedAt());
        entity.setUpdatedAt(file.getUpdatedAt());
        return entity;
    }
}
