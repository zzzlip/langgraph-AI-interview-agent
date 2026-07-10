package com.aiinterview.platform.application.security;

import com.aiinterview.platform.application.file.FileException;
import com.aiinterview.platform.domain.file.FileType;
import java.util.Locale;

/**
 * V1 文件内容安全检查。
 */
public class FileContentInspector {

    private static final int FILE_TYPE_NOT_SUPPORTED = 72002;

    public FileContentInspectionResult inspect(String originalName, String declaredMimeType, byte[] content) {
        String extension = extensionOf(originalName);
        FileType fileType = FileType.fromCode(extension)
                .orElseThrow(() -> new FileException(FILE_TYPE_NOT_SUPPORTED, "文件类型不支持"));
        if (!fileType.getMimeType().equals(declaredMimeType)) {
            throw new FileException(FILE_TYPE_NOT_SUPPORTED, "文件类型不支持");
        }
        if (!matchesMagic(fileType, content)) {
            throw new FileException(FILE_TYPE_NOT_SUPPORTED, "文件类型不支持");
        }
        return new FileContentInspectionResult(fileType);
    }

    private static String extensionOf(String originalName) {
        int index = originalName == null ? -1 : originalName.lastIndexOf('.');
        if (index < 0 || index == originalName.length() - 1) {
            return "";
        }
        return originalName.substring(index + 1).toLowerCase(Locale.ROOT);
    }

    private static boolean matchesMagic(FileType fileType, byte[] content) {
        if (fileType == FileType.JSON) {
            return looksLikeJson(content);
        }
        if (content == null || content.length < 4) {
            return false;
        }
        return switch (fileType) {
            case PDF -> startsWith(content, "%PDF".getBytes());
            case PNG -> startsWith(content, new byte[]{(byte) 0x89, 'P', 'N', 'G'});
            case JPG, JPEG -> startsWith(content, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
            case DOC -> startsWith(content, new byte[]{(byte) 0xD0, (byte) 0xCF, 0x11, (byte) 0xE0});
            case DOCX -> startsWith(content, new byte[]{'P', 'K', 0x03, 0x04});
            case MP3 -> startsWith(content, "ID3".getBytes()) || startsWith(content, new byte[]{(byte) 0xFF, (byte) 0xFB});
            case MP4 -> containsAt(content, "ftyp".getBytes(), 4);
            case WEBM -> startsWith(content, new byte[]{0x1A, 0x45, (byte) 0xDF, (byte) 0xA3});
            case JSON -> looksLikeJson(content);
        };
    }

    private static boolean looksLikeJson(byte[] content) {
        if (content == null || content.length < 2) {
            return false;
        }
        for (byte value : content) {
            if (!Character.isWhitespace(value)) {
                return value == '{' || value == '[';
            }
        }
        return false;
    }

    private static boolean startsWith(byte[] content, byte[] prefix) {
        if (content.length < prefix.length) {
            return false;
        }
        for (int i = 0; i < prefix.length; i++) {
            if (content[i] != prefix[i]) {
                return false;
            }
        }
        return true;
    }

    private static boolean containsAt(byte[] content, byte[] expected, int offset) {
        if (content.length < offset + expected.length) {
            return false;
        }
        for (int i = 0; i < expected.length; i++) {
            if (content[offset + i] != expected[i]) {
                return false;
            }
        }
        return true;
    }
}
