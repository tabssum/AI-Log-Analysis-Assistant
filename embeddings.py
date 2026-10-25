from functools import lru_cache
from typing import Sequence

from httpx import HTTPError
from langchain_core.exceptions import ModelError
from langchain_ollama import OllamaEmbeddings
from ollama import RequestError, ResponseError


DEFAULT_MODEL_NAME = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama cannot generate log embeddings."""


@lru_cache(maxsize=2)
def _get_model(model_name: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=model_name,
        base_url=DEFAULT_OLLAMA_BASE_URL,
    )


def generate_embeddings(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[list[float]]:
    """Generate Ollama embeddings for log chunks."""
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings")

    try:
        vectors = _get_model(model_name).embed_documents(list(texts))
    except (HTTPError, ModelError, RequestError, ResponseError) as error:
        raise OllamaEmbeddingError(
            f"Ollama could not generate document embeddings: {error}. "
            f"Confirm Ollama is running and the '{model_name}' model is installed."
        ) from error
    return [[float(value) for value in vector] for vector in vectors]


def generate_query_embedding(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[float]:
    """Generate an Ollama embedding for semantic search."""
    if not text.strip():
        raise ValueError("text must be a non-empty string")

    try:
        vector = _get_model(model_name).embed_query(text)
    except (HTTPError, ModelError, RequestError, ResponseError) as error:
        raise OllamaEmbeddingError(
            f"Ollama could not generate a query embedding: {error}. "
            f"Confirm Ollama is running and the '{model_name}' model is installed."
        ) from error
    return [float(value) for value in vector]
