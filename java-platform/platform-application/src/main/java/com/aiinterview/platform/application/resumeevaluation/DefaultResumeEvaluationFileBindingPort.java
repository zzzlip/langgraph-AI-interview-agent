package com.aiinterview.platform.application.resumeevaluation;

import com.aiinterview.platform.application.resumeevaluation.dto.ResumeEvaluationInputFile;
import com.aiinterview.platform.application.resumeevaluation.port.ResumeEvaluationFileBindingPort;
import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.file.FileType;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * 默认简历评估输入文件校验端口。
 */
public class DefaultResumeEvaluationFileBindingPort implements ResumeEvaluationFileBindingPort {

    private static final int FILE_NOT_FOUND = 72001;
    private static final int FILE_BIND_NOT_ALLOWED = 72006;
    private static final Set<FileType> RESUME_TYPES = EnumSet.of(FileType.PDF, FileType.DOC, FileType.DOCX);

    private final FileRepository fileRepository;

    public DefaultResumeEvaluationFileBindingPort(FileRepository fileRepository) {
        this.fileRepository = Objects.requireNonNull(fileRepository, "fileRepository 不能为空");
    }

    @Override
    public ResumeEvaluationInputFile validateInputResume(String userId, String resumeFileId) {
        File file = fileRepository.findById(resumeFileId)
                .orElseThrow(() -> new ResumeEvaluationException(FILE_NOT_FOUND, "简历文件不存在"));
        try {
            file.assertOwnedBy(userId);
        } catch (RuntimeException ex) {
            throw new ResumeEvaluationException(FILE_NOT_FOUND, "简历文件不存在", ex);
        }
        if (!file.canBindAsInput() || file.getFilePurpose() != FilePurpose.RESUME || !RESUME_TYPES.contains(file.getFileType())) {
            throw new ResumeEvaluationException(FILE_BIND_NOT_ALLOWED, "文件不可用于简历评估");
        }
        return new ResumeEvaluationInputFile(
                file.getId(),
                file.getOriginalName(),
                file.getObjectKey(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode()
        );
    }
}
