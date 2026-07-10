package com.aiinterview.platform.application.security;

import com.aiinterview.platform.domain.file.FileType;

/**
 * 文件内容检查结果。
 */
public record FileContentInspectionResult(FileType fileType) {
}
