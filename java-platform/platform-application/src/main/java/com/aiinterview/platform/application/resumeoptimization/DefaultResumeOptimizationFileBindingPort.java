package com.aiinterview.platform.application.resumeoptimization;

import com.aiinterview.platform.application.resumeoptimization.dto.ResumeOptimizationInputFile;
import com.aiinterview.platform.application.resumeoptimization.port.ResumeOptimizationFileBindingPort;
import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.file.FileType;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * 默认简历优化输入文件校验端口。
 */
public class DefaultResumeOptimizationFileBindingPort implements ResumeOptimizationFileBindingPort {

    private static final int FILE_NOT_FOUND = 72001;
    private static final int FILE_BIND_NOT_ALLOWED = 72006;
    private static final Set<FileType> RESUME_TYPES = EnumSet.of(FileType.PDF, FileType.DOC, FileType.DOCX);
    private static final Set<FileType> AVATAR_TYPES = EnumSet.of(FileType.PNG, FileType.JPG, FileType.JPEG);

    private final FileRepository fileRepository;

    public DefaultResumeOptimizationFileBindingPort(FileRepository fileRepository) {
        this.fileRepository = Objects.requireNonNull(fileRepository, "fileRepository 不能为空");
    }

    @Override
    public ResumeOptimizationInputFile validateInputResume(String userId, String resumeFileId) {
        File file = requireOwnedFile(userId, resumeFileId, "简历文件不存在");
        if (!file.canBindAsInput() || file.getFilePurpose() != FilePurpose.RESUME || !RESUME_TYPES.contains(file.getFileType())) {
            throw new ResumeOptimizationException(FILE_BIND_NOT_ALLOWED, "文件不可用于简历优化");
        }
        return toInputFile(file);
    }

    @Override
    public ResumeOptimizationInputFile validateOptionalAvatar(String userId, String avatarFileId) {
        File file = requireOwnedFile(userId, avatarFileId, "头像文件不存在");
        if (!file.canBindAsInput() || file.getFilePurpose() != FilePurpose.AVATAR || !AVATAR_TYPES.contains(file.getFileType())) {
            throw new ResumeOptimizationException(FILE_BIND_NOT_ALLOWED, "头像文件不可用于简历优化");
        }
        return toInputFile(file);
    }

    private File requireOwnedFile(String userId, String fileId, String message) {
        File file = fileRepository.findById(fileId)
                .orElseThrow(() -> new ResumeOptimizationException(FILE_NOT_FOUND, message));
        try {
            file.assertOwnedBy(userId);
        } catch (RuntimeException ex) {
            throw new ResumeOptimizationException(FILE_NOT_FOUND, message, ex);
        }
        return file;
    }

    private static ResumeOptimizationInputFile toInputFile(File file) {
        return new ResumeOptimizationInputFile(
                file.getId(),
                file.getOriginalName(),
                file.getObjectKey(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode()
        );
    }
}
