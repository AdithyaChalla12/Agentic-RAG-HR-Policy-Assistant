"""Index a folder of documents into Pinecone.

Usage:
    python -m scripts.ingest_kb                 # data/sample_kb
    python -m scripts.ingest_kb path/to/folder
"""
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.vectorstore import upsert_chunks
from app.services.ingestion import SUPPORTED, chunk_documents, load_file


def main(folder: Path) -> None:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        sys.exit(f"No {sorted(SUPPORTED)} files in {folder}")
    docs = [d for f in files for d in load_file(f)]
    chunks = chunk_documents(docs)
    n = upsert_chunks(chunks)
    print(f"Indexed {len(files)} files -> {len(chunks)} chunks -> {n} vectors")
    for f in files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    configure_logging()
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(get_settings().sample_kb_dir))
