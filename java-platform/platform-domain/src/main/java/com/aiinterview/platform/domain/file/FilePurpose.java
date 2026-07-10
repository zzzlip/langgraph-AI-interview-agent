package com.aiinterview.platform.domain.file;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Optional;
import java.util.Set;

/**
 * V1 文件用途。
 */
public enum FilePurpose {
    RESUME("resume", true, 20 * 1024 * 1024L, FileType.PDF, FileType.DOC, FileType.DOCX),
    AVATAR("avatar", true, 20 * 1024 * 1024L, FileType.PNG, FileType.JPG, FileType.JPEG),
    INTERVIEW_AUDIO("interview_audio", true, 50 * 1024 * 1024L, FileType.MP3),
    INTERVIEW_VIDEO("interview_video", true, 50 * 1024 * 1024L, FileType.MP4, FileType.WEBM),
    RADAR_CHART("radar_chart", false, 20 * 1024 * 1024L, FileType.PNG),
    EVALUATION_REPORT("evaluation_report", false, 20 * 1024 * 1024L, FileType.DOCX),
    INTERVIEW_REPORT("interview_report", false, 20 * 1024 * 1024L, FileType.DOCX, FileType.PDF),
    OPTIMIZED_RESUME("optimized_resume", false, 20 * 1024 * 1024L, FileType.PDF),
    ALGORITHM_TEST_PROBLEM_SET("algorithm_test_problem_set", false, 20 * 1024 * 1024L, FileType.JSON),
    ALGORITHM_TEST_RESULT("algorithm_test_result", false, 20 * 1024 * 1024L, FileType.JSON);

    private final String code;
    private final boolean userUpload;
    private final long maxSizeBytes;
    private final Set<FileType> allowedTypes;

    FilePurpose(String code, boolean userUpload, long maxSizeBytes, FileType firstType, FileType... otherTypes) {
        this.code = code;
        this.userUpload = userUpload;
        this.maxSizeBytes = maxSizeBytes;
        this.allowedTypes = EnumSet.of(firstType, otherTypes);
    }

    public String getCode() {
        return code;
    }

    public boolean isUserUpload() {
        return userUpload;
    }

    public long getMaxSizeBytes() {
        return maxSizeBytes;
    }

    /**
     * 判断当前用途是否允许指定文件类型；未知类型默认拒绝。
     */
    public boolean allows(FileType fileType) {
        return fileType != null && allowedTypes.contains(fileType);
    }

    /**
     * 按外部 code 查找文件用途；未知值保持拒绝语义。
     */
    public static Optional<FilePurpose> fromCode(String code) {
        return Arrays.stream(values())
                .filter(purpose -> purpose.code.equals(code))
                .findFirst();
    }
}
