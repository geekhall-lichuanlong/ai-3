from pathlib import Path

from pypdf import PdfReader

from app.chunking import ParsedSection


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def parse_document(path: Path, filename: str) -> list[ParsedSection]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type: {ext}")
    if ext == ".pdf":
        return parse_pdf(path)
    return parse_text(path)


def parse_pdf(path: Path) -> list[ParsedSection]:
    reader = PdfReader(str(path))
    sections: list[ParsedSection] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(ParsedSection(text=text, page=index))
    return sections


def parse_text(path: Path) -> list[ParsedSection]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    sections: list[ParsedSection] = []
    buffer: list[str] = []
    start_line = 1
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if not buffer:
                start_line = line_number
            buffer.append(line)
            continue
        if buffer:
            sections.append(
                ParsedSection(
                    text="\n".join(buffer),
                    start_line=start_line,
                    end_line=line_number - 1,
                )
            )
            buffer = []
    if buffer:
        sections.append(
            ParsedSection(
                text="\n".join(buffer),
                start_line=start_line,
                end_line=len(lines),
            )
        )
    return sections

