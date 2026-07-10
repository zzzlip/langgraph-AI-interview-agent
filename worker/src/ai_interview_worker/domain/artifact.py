"""Worker 输出产物领域模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputArtifact:
    """Worker 写入对象存储后的产物引用。"""

    object_key: str
    file_type: str
    file_purpose: str
    mime_type: str
    file_size: int
    display_name: str

    def __post_init__(self) -> None:
        if self.file_size <= 0:
            raise ValueError("file_size must be positive")
        if _contains_path_separator_or_control_char(self.display_name):
            raise ValueError("display_name must not contain path separators or control characters")
        if not self.mime_type:
            object.__setattr__(self, "mime_type", "application/octet-stream")


def _contains_path_separator_or_control_char(value: str) -> bool:
    return "/" in value or "\\" in value or any(ord(char) < 32 for char in value)
