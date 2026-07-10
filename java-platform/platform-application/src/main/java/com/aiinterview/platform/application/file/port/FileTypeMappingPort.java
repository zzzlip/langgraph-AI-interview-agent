package com.aiinterview.platform.application.file.port;

import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileType;

/**
 * 文件类型与用途映射端口。
 */
public interface FileTypeMappingPort {

    boolean supports(FilePurpose purpose, FileType type);
}
