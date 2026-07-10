"""简历优化旧原型可复用文本处理能力的受控适配入口。"""

from io import BytesIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_resume_text(file_bytes: bytes, file_type: str) -> str:
    """从文件 bytes 提取文本，不接收本地路径。"""
    if not isinstance(file_bytes, bytes):
        raise TypeError("file_bytes must be bytes")
    if not file_bytes:
        raise ValueError("empty resume content")
    if file_type not in {"pdf", "doc", "docx", "txt"}:
        raise ValueError("unsupported resume file type")
    if file_type == "pdf":
        return _extract_pdf_text(file_bytes)
    if file_type == "docx":
        return _extract_docx_text(file_bytes)
    return _decode_text(file_bytes)


def _extract_pdf_text(file_bytes: bytes) -> str:
    if not file_bytes.lstrip().startswith(b"%PDF"):
        return _decode_text(file_bytes)
    try:
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except PdfReadError:
        return _decode_text(file_bytes)


def _extract_docx_text(file_bytes: bytes) -> str:
    try:
        with ZipFile(BytesIO(file_bytes)) as archive:
            document_xml = archive.read("word/document.xml")
    except (BadZipFile, KeyError):
        return _decode_text(file_bytes)

    root = ElementTree.fromstring(document_xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")
