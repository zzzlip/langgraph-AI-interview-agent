"""简历评估旧能力受控适配入口。"""

import re
from dataclasses import dataclass
from io import BytesIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SECTION_HEADERS = {
    "教育经历": "school",
    "教育背景": "school",
    "教育": "school",
    "项目经历": "program",
    "项目经验": "program",
    "项目": "program",
    "工作经历": "work",
    "工作经验": "work",
    "实习经历": "work",
    "专业技能": "technology_stack",
    "技能栈": "technology_stack",
    "技能": "technology_stack",
    "荣誉奖项": "awards",
    "获奖情况": "awards",
    "奖项": "awards",
    "个人优势": "technology_stack",
    "志愿者服务经历": "school_work",
    "社团/组织经历": "school_work",
}

HEADER_PATTERN = re.compile(
    r"(?P<header>"
    + "|".join(re.escape(header) for header in sorted(SECTION_HEADERS, key=len, reverse=True))
    + r")\s*[：:\n]"
)


@dataclass(frozen=True)
class ParsedResume:
    """解析后的简历内容。"""

    full_text: str
    sections: dict[str, str]


def parse_resume(file_bytes: bytes, file_type: str) -> ParsedResume:
    """从文件 bytes 解析简历文本，不接收本地路径。"""
    if not file_bytes:
        raise ValueError("empty resume content")
    if file_type not in {"pdf", "doc", "docx", "txt"}:
        raise ValueError("unsupported resume file type")
    text = _extract_text(file_bytes, file_type).strip()
    if not text or not any(not char.isspace() and ord(char) >= 32 for char in text):
        raise ValueError("empty resume text")
    return ParsedResume(
        full_text=text,
        sections=_extract_sections(text),
    )


def _extract_text(file_bytes: bytes, file_type: str) -> str:
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


def _extract_sections(text: str) -> dict[str, str]:
    sections = {
        "raw_text": text,
        "program": "",
        "work": "",
        "school": "",
        "technology_stack": "",
        "awards": "",
        "school_work": "",
    }
    buffers: dict[str, list[str]] = {field: [] for field in sections if field != "raw_text"}
    matches = list(HEADER_PATTERN.finditer(text))
    if not matches:
        return sections

    for index, match in enumerate(matches):
        header = match.group("header")
        field = SECTION_HEADERS[header]
        content_start = match.end()
        content_end = matches[index + 1].start() if index < len(matches) - 1 else len(text)
        content = text[content_start:content_end].strip(" \n\r\t：:")
        if content:
            buffers[field].append(content)

    for field, lines in buffers.items():
        sections[field] = "\n".join(lines).strip()
    return sections
