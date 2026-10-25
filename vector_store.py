from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from chunker import LogChunk
from embeddings import (
    DEFAULT_MODEL_NAME,
    generate_embeddings,
    generate_query_embedding,
)


DEFAULT_COLLECTION_NAME = "log_chunks_ollama_nomic_embed_text"


@dataclass(frozen=True)
class VectorSearchResult:
    text: str
    metadata: dict[str, str | int]
    distance: float


class ChromaVectorStore:
    """Persist and search log-chunk embeddings in a local ChromaDB collection."""

    def __init__(
        self,
        persistence_path: str | Path = ".cache/chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        import chromadb

        self.model_name = model_name
        client = chromadb.PersistentClient(path=str(persistence_path))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: Sequence[LogChunk],
        source: str,
        source_id: str | None = None,
    ) -> int:
        """Embed and upsert chunks, using source and chunk index as stable IDs."""
        if not source.strip():
            raise ValueError("source must be a non-empty string")
        if source_id is not None and not source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not chunks:
            return 0

        chunk_source_id = source_id or source
        documents = [chunk.text for chunk in chunks]
        embeddings = generate_embeddings(documents, model_name=self.model_name)
        self.collection.upsert(
            ids=[f"{chunk_source_id}:{chunk.chunk_index}" for chunk in chunks],
            documents=documents,
            metadatas=[
                {
                    "source": source,
                    "source_id": chunk_source_id,
                    "chunk_index": chunk.chunk_index,
                    "start_line_number": chunk.start_line_number,
                    "end_line_number": chunk.end_line_number,
                    "error_line_numbers": ",".join(
                        str(line_number) for line_number in chunk.error_line_numbers
                    ),
                }
                for chunk in chunks
            ],
            embeddings=embeddings,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        source: str | None = None,
        source_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Return the nearest stored chunks for a natural-language query."""
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if n_results < 1:
            raise ValueError("n_results must be greater than zero")
        if source_id is not None and not source_id.strip():
            raise ValueError("source_id must be a non-empty string")

        query_embedding = generate_query_embedding(
            query,
            model_name=self.model_name,
        )
        filters = {"source_id": source_id} if source_id is not None else None
        if filters is None and source is not None:
            filters = {"source": source}
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            VectorSearchResult(
                text=document,
                metadata=metadata,
                distance=distance,
            )
            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
            )
        ]

    def count(self) -> int:
        """Return the number of stored log chunks."""
        return self.collection.count()

    def has_source_chunks(self, source_id: str, expected_count: int) -> bool:
        """Check whether all chunks for an uploaded source are already indexed."""
        if not source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if expected_count < 1:
            raise ValueError("expected_count must be greater than zero")

        stored_chunks = self.collection.get(
            where={"source_id": source_id},
            include=[],
        )
        return len(stored_chunks["ids"]) == expected_count
