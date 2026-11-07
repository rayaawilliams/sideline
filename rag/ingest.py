"""Build the vector database from the documents in data/docs.

Usage: python -m rag.ingest [--reset]
"""
import argparse
import time

from .config import load_settings
from .documents import chunk_pages, load_pages
from .embeddings import make_embedder
from .store import VectorStore


def ingest(settings, embedder=None, reset: bool = False) -> int:
    embedder = embedder or make_embedder(settings)
    if reset:
        VectorStore.delete(settings.db_dir, settings.collection)
    store = VectorStore(settings.db_dir, settings.collection, embedder.name)

    pages = load_pages(settings.docs_dir)
    chunks = chunk_pages(pages, settings.chunk_chars, settings.chunk_overlap)
    if not chunks:
        print(f"No documents found in {settings.docs_dir}. Add PDF, .txt or .md files first.")
        return 0
    vectors = embedder.embed([c.text for c in chunks])
    store.add(chunks, vectors)
    sources = sorted({c.source for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(pages)} pages in {len(sources)} files:")
    for source in sources:
        print(f"  {source}")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete the database and rebuild it")
    parser.add_argument("--embedder", help="sentence-transformers (default) or hashing")
    args = parser.parse_args()
    started = time.perf_counter()
    ingest(load_settings(embedder=args.embedder), reset=args.reset)
    print(f"Done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
