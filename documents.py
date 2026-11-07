"""Load documents from disk and split them into overlapping chunks."""
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Page:
    text: str
    source: str  # file name
    page: int | None  # 1-based page number for PDFs, None for text files


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int | None
    index: int  # position of the chunk within its page

    @property
    def id(self) -> str:
        return f"{self.source}::p{self.page or 0}::c{self.index}"


def load_pages(docs_dir: str) -> list:
    """Read every supported file under docs_dir, one Page per PDF page or text file."""
    pages = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() not in SUPPORTED or path.name.lower() == "readme.md":
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            for number, pdf_page in enumerate(PdfReader(path).pages, start=1):
                text = (pdf_page.extract_text() or "").strip()
                if text:
                    pages.append(Page(text, path.name, number))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                pages.append(Page(text, path.name, None))
    return pages


def chunk_text(text: str, max_chars: int, overlap: int) -> list:
    """Split text into pieces of at most max_chars, preferring paragraph and sentence breaks.

    Each chunk after the first starts with the last `overlap` characters of the previous
    one, so a sentence cut at a boundary is still readable in full somewhere.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = text.strip()
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Back up to the nearest paragraph, sentence or word break in the second half.
            window = text[start:end]
            for separator in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(separator)
                if cut > max_chars // 2:
                    end = start + cut + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_pages(pages: list, max_chars: int, overlap: int) -> list:
    chunks = []
    for page in pages:
        for index, piece in enumerate(chunk_text(page.text, max_chars, overlap)):
            chunks.append(Chunk(piece, page.source, page.page, index))
    return chunks
