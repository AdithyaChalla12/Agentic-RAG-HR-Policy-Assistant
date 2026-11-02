"""Load -> split -> tag. Turns raw files into chunks ready for embedding.

Every chunk gets:
  - metadata.source   : file name (shown as a citation)
  - metadata.section  : heading path for Markdown ("Leave > Parental leave")
  - metadata.page     : page number for PDFs
  - a deterministic id: "<doc-slug>#<n>" so re-ingesting a file overwrites it
"""
import re
from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def doc_slug(source: str) -> str:
    """'Leave Policy v2.pdf' -> 'leave-policy-v2-pdf' (stable Pinecone id prefix)."""
    return re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")


def load_file(path: Path) -> list[Document]:
    """Read one file into LangChain Documents (one per PDF page, one otherwise)."""
    suffix = path.suffix.lower()
    source = path.name
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=text, metadata={"source": source, "page": i + 1})
            for i, page in enumerate(reader.pages)
            if (text := (page.extract_text() or "").strip())
        ]
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": source, "format": suffix[1:]})]
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": source})]
    raise ValueError(f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED)}")


def _split_one(doc: Document) -> list[Document]:
    """Structure-aware for Markdown (split on headings first), recursive for the rest."""
    if doc.metadata.get("format") == "md":
        sections = _md_splitter.split_text(doc.page_content)
        for s in sections:
            heads = [s.metadata.pop(k) for k in ("h1", "h2", "h3") if k in s.metadata]
            s.metadata = {**doc.metadata, "section": " > ".join(heads)}
        return _text_splitter.split_documents(sections)
    return _text_splitter.split_documents([doc])


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    """Split and assign deterministic ids: '<doc-slug>#<n>' numbered per source file."""
    chunks: list[Document] = []
    counters: dict[str, int] = {}
    for doc in docs:
        for chunk in _split_one(doc):
            slug = doc_slug(chunk.metadata["source"])
            n = counters.get(slug, 0)
            counters[slug] = n + 1
            chunk.id = f"{slug}#{n}"
            chunk.metadata["chunk_id"] = chunk.id
            chunks.append(chunk)
    return chunks
