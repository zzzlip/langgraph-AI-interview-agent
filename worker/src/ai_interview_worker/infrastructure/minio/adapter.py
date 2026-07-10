"""MinIO 对象适配器（只供 infrastructure 层 import minio SDK）。"""

from collections.abc import Callable
from datetime import datetime
from mimetypes import guess_type

from ai_interview_worker.application.port.file_ports import (
    ReadInputResult,
    WriteArtifactResult,
    artifact_write_failure,
    input_not_found_failure,
    minio_connect_failure,
)
from ai_interview_worker.domain.artifact import OutputArtifact


class MinioFileAdapter:
    """MinIO 文件操作适配器。"""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        from minio import Minio

        self.bucket = bucket
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def get_object(self, object_key: str) -> ReadInputResult:
        """读取 object key 对应文件内容。"""
        try:
            response = self.client.get_object(self.bucket, object_key)
            data = response.read()
            response.close()
            response.release_conn()
            mime_type = guess_type(object_key)[0] or "application/octet-stream"
            return ReadInputResult(success=True, data=data, mime_type=mime_type)
        except Exception as ex:
            error_msg = str(ex)
            if "Not Found" in error_msg or "NoSuchKey" in error_msg or object_key in error_msg:
                return ReadInputResult(success=False, failure=input_not_found_failure())
            return ReadInputResult(success=False, failure=minio_connect_failure())

    @staticmethod
    def _mime_to_file_type(mime_type: str, display_name: str) -> str:
        """从 MIME 和展示名推导文件类型扩展名。"""
        ext = display_name.rsplit(".", 1)[-1].lower() if "." in display_name else ""
        if ext and ext not in ("bin", "unknown"):
            return ext
        if "/" in mime_type:
            subtype = mime_type.rsplit("/", 1)[-1]
            mime_ext_map = {
                "vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                "vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                "plain": "txt",
                "octet-stream": "bin",
            }
            return mime_ext_map.get(subtype, subtype)
        return mime_type

    def put_object(
        self,
        *,
        object_key: str,
        data: bytes,
        mime_type: str,
        display_name: str,
        clock: Callable[[], datetime],
    ) -> WriteArtifactResult:
        """写入产物到 MinIO。"""
        import io

        try:
            self.client.put_object(
                self.bucket,
                object_key,
                io.BytesIO(data),
                length=len(data),
                content_type=mime_type,
            )
            artifact = OutputArtifact(
                object_key=object_key,
                file_type=self._mime_to_file_type(mime_type, display_name),
                file_purpose=object_key.split("/")[-2] if "/" in object_key else "output",
                mime_type=mime_type,
                file_size=len(data),
                display_name=display_name,
            )
            return WriteArtifactResult(success=True, artifact=artifact)
        except Exception:
            return WriteArtifactResult(success=False, failure=artifact_write_failure())
