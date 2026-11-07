"""Vector database: stores chunks with their vectors and finds the closest ones to a question."""
from dataclasses import dataclass

from .documents import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, 1.0 = identical direction


class VectorStore:
    """Thin wrapper around a persistent Chroma collection.

    Vectors always come from our own Embedder, so Chroma never downloads its default model.
    """

    def __init__(self, db_dir: str, collection: str, embedder_name: str):
        import chromadb

        self._client = chromadb.PersistentClient(path=db_dir)
        self._name = collection
        self._collection = self._client.get_or_create_collection(
            collection,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedder": embedder_name},
        )
        stored = (self._collection.metadata or {}).get("embedder")
        if stored and stored != embedder_name:
            raise ValueError(
                f"Collection was built with embedder '{stored}' but '{embedder_name}' is in use. "
                "Re-run ingest with --reset."
            )

    @staticmethod
    def delete(db_dir: str, collection: str) -> None:
        """Remove a collection if it exists, e.g. before re-indexing with a different embedder."""
        import chromadb

        client = chromadb.PersistentClient(path=db_dir)
        if collection in [c.name for c in client.list_collections()]:
            client.delete_collection(collection)

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list, vectors: list, batch_size: int = 500) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=vectors[start:start + batch_size],
                documents=[c.text for c in batch],
                metadatas=[
                    {"source": c.source, "page": c.page or 0, "index": c.index} for c in batch
                ],
            )

    def query(self, vector: list, top_k: int) -> list:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector], n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0],
        ):
            chunk = Chunk(text, meta["source"], meta["page"] or None, meta["index"])
            hits.append(Hit(chunk, 1.0 - distance))
        return hits
