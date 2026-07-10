package com.aiinterview.platform.application.mockinterview;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerFile;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewResumeFile;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewFileBindingPort;
import com.aiinterview.platform.domain.file.File;
import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.file.FileType;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * 默认模拟面试回答文件校验端口。
 */
public class DefaultMockInterviewFileBindingPort implements MockInterviewFileBindingPort {

    private static final int FILE_NOT_FOUND = 72001;
    private static final int FILE_BIND_NOT_ALLOWED = 72006;
    private static final Set<FileType> RESUME_TYPES = EnumSet.of(FileType.PDF, FileType.DOC, FileType.DOCX);

    private final FileRepository fileRepository;

    public DefaultMockInterviewFileBindingPort(FileRepository fileRepository) {
        this.fileRepository = Objects.requireNonNull(fileRepository, "fileRepository 不能为空");
    }

    @Override
    public MockInterviewResumeFile validateResumeFile(String userId, String resumeFileId) {
        File file = ownedFile(userId, resumeFileId, "简历文件不存在");
        if (!file.canBindAsInput()
                || file.getFilePurpose() != FilePurpose.RESUME
                || !RESUME_TYPES.contains(file.getFileType())) {
            throw new MockInterviewException(FILE_BIND_NOT_ALLOWED, "文件不可用于模拟面试");
        }
        return new MockInterviewResumeFile(
                file.getId(),
                file.getOriginalName(),
                file.getObjectKey(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode(),
                file.getMimeType()
        );
    }

    @Override
    public MockInterviewAnswerFile validateAnswerFile(String userId, String answerFileId) {
        File file = ownedFile(userId, answerFileId, "回答文件不存在");
        if (!file.canBindAsInput()
                || (file.getFilePurpose() != FilePurpose.INTERVIEW_AUDIO
                && file.getFilePurpose() != FilePurpose.INTERVIEW_VIDEO)) {
            throw new MockInterviewException(FILE_BIND_NOT_ALLOWED, "回答文件不可用于模拟面试");
        }
        return new MockInterviewAnswerFile(
                file.getId(),
                file.getObjectKey(),
                file.getFileType().getCode(),
                file.getFilePurpose().getCode(),
                file.getFilePurpose() == FilePurpose.INTERVIEW_AUDIO ? "audio" : "video",
                file.getMimeType()
        );
    }

    private File ownedFile(String userId, String fileId, String notFoundMessage) {
        File file = fileRepository.findById(fileId)
                .orElseThrow(() -> new MockInterviewException(FILE_NOT_FOUND, notFoundMessage));
        try {
            file.assertOwnedBy(userId);
        } catch (RuntimeException ex) {
            throw new MockInterviewException(FILE_NOT_FOUND, notFoundMessage, ex);
        }
        return file;
    }
}
