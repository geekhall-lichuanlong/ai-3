from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSection:
    text: str
    page: int | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int | None = None
    start_line: int | None = None
    end_line: int | None = None


def chunk_sections(
    sections: list[ParsedSection],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    for section in sections:
        text = " ".join(section.text.split())
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        page=section.page,
                        start_line=section.start_line,
                        end_line=section.end_line,
                    )
                )
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
    return chunks

