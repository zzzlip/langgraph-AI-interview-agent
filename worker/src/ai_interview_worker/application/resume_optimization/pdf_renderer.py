"""简历优化 PDF 渲染。"""

import base64
import html
import importlib
import re
from io import BytesIO
from typing import Any, cast


def render_resume_pdf(markdown: str, avatar: bytes | None) -> bytes:
    """将优化后 Markdown 渲染为可下载 PDF bytes。"""
    html_document = _build_html_document(markdown, avatar)
    weasyprint = importlib.import_module("weasyprint")
    html_cls = cast(Any, weasyprint.HTML)
    pdf = cast(bytes, html_cls(string=html_document).write_pdf())
    if not pdf.startswith(b"%PDF-") or len(pdf) <= 1024:
        raise ValueError("rendered pdf is invalid")
    return pdf


def _build_html_document(markdown: str, avatar: bytes | None) -> str:
    avatar_html = ""
    body_class = " with-avatar" if avatar is not None else ""
    if avatar is not None:
        avatar_html = f'<img class="avatar" src="{_avatar_data_uri(avatar)}" alt="avatar" />'
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\" />"
        f"<style>{_stylesheet()}</style></head>"
        f"<body class=\"resume{body_class}\">{avatar_html}<main>{_markdown_to_html_body(markdown)}</main></body></html>"
    )


def _avatar_data_uri(avatar: bytes) -> str:
    image_module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    image_draw = importlib.import_module("PIL.ImageDraw")
    with cast(Any, image_module).open(BytesIO(avatar)) as image:
        normalized = image.convert("RGBA")
        resized = cast(Any, image_ops).fit(normalized, (150, 150))
        mask = cast(Any, image_module).new("L", (150, 150), 0)
        draw = cast(Any, image_draw).Draw(mask)
        draw.ellipse((0, 0, 150, 150), fill=255)
        resized.putalpha(mask)
        output = BytesIO()
        resized.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _markdown_to_html_body(markdown: str) -> str:
    parts: list[str] = []
    in_list = False
    in_table = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            parts.append("</tbody></table>")
            in_table = False

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            close_list()
            close_table()
            continue
        if line == "---":
            close_list()
            close_table()
            parts.append("<hr />")
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            close_list()
            close_table()
            level = min(len(heading_match.group(1)), 3)
            parts.append(f"<h{level}>{_inline_format(heading_match.group(2))}</h{level}>")
            continue
        if line.startswith(("- ", "* ", "• ")):
            close_table()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{_inline_format(line[2:].strip())}</li>")
            continue
        if _is_table_row(line):
            close_list()
            if _is_table_separator(line):
                continue
            if not in_table:
                parts.append("<table><tbody>")
                in_table = True
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            parts.append("<tr>" + "".join(f"<td>{_inline_format(cell)}</td>" for cell in cells) + "</tr>")
            continue

        close_list()
        close_table()
        parts.append(f"<p>{_inline_format(line)}</p>")

    close_list()
    close_table()
    return "\n".join(parts)


def _inline_format(value: str) -> str:
    escaped = html.escape(value, quote=True)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _stylesheet() -> str:
    return """
@page { size: A4; margin: 20mm 18mm; }
body {
  color: #111827;
  font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS", sans-serif;
  font-size: 11px;
  line-height: 1.55;
}
main { position: relative; }
.with-avatar main { padding-top: 6mm; }
.avatar {
  position: fixed;
  top: 0;
  right: 0;
  width: 30mm;
  height: 30mm;
  border-radius: 50%;
}
h1, h2, h3 {
  color: #0f172a;
  margin: 0 0 3mm;
  page-break-after: avoid;
}
h1 { font-size: 22px; }
h2 { font-size: 16px; }
h3 { font-size: 13px; }
hr {
  border: 0;
  border-top: 1px solid #cbd5e1;
  margin: 0 0 4mm;
}
p { margin: 0 0 3mm; }
ul { margin: 0 0 3mm 5mm; padding: 0; }
li { margin: 0 0 1.5mm; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 4mm;
}
td {
  border: 0.5px solid #cbd5e1;
  padding: 2mm;
  vertical-align: top;
}
strong { font-weight: 700; }
"""
