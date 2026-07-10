package com.aiinterview.platform.infrastructure.persistence.file;

import com.aiinterview.platform.domain.file.FileBindType;
import com.aiinterview.platform.domain.file.FilePurpose;
import com.aiinterview.platform.domain.file.FileStatus;
import com.aiinterview.platform.domain.file.FileType;

/**
 * 文件枚举转换。
 */
final class FileEnumMapper {

    private FileEnumMapper() {
    }

    static FileType typeFromCode(String code) {
        return FileType.fromCode(code).orElseThrow(() -> new IllegalArgumentException("未知文件类型: " + code));
    }

    static FilePurpose purposeFromCode(String code) {
        return FilePurpose.fromCode(code).orElseThrow(() -> new IllegalArgumentException("未知文件用途: " + code));
    }

    static FileStatus statusFromCode(String code) {
        return FileStatus.fromCode(code).orElseThrow(() -> new IllegalArgumentException("未知文件状态: " + code));
    }

    static FileBindType bindTypeFromCode(String code) {
        for (FileBindType bindType : FileBindType.values()) {
            if (bindType.getCode().equals(code)) {
                return bindType;
            }
        }
        throw new IllegalArgumentException("未知文件绑定类型: " + code);
    }
}
