from typing import Sequence

from google.genai.errors import APIError

from gemini_client import GeminiConfigurationError, create_gemini_client


DEFAULT_MODEL_NAME = "gemini-embedding-2"


class GeminiEmbeddingError(RuntimeError):
    """Raised when Gemini cannot generate log embeddings."""


def generate_embeddings(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[list[float]]:
    """Generate Gemini embeddings for log chunks."""
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings")

    try:
        response = create_gemini_client().models.embed_content(
            model=model_name,
            contents=[f"title: none | text: {text}" for text in texts],
        )
    except (APIError, GeminiConfigurationError) as error:
        raise GeminiEmbeddingError(
            f"Gemini could not generate document embeddings: {error}"
        ) from error

    if response.embeddings is None or len(response.embeddings) != len(texts):
        raise GeminiEmbeddingError(
            "Gemini returned an unexpected number of document embeddings."
        )
    if any(embedding.values is None for embedding in response.embeddings):
        raise GeminiEmbeddingError("Gemini returned an empty document embedding.")
    return [
        [float(value) for value in embedding.values]
        for embedding in response.embeddings
        if embedding.values is not None
    ]


def generate_query_embedding(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[float]:
    """Generate a Gemini embedding for semantic search."""
    if not text.strip():
        raise ValueError("text must be a non-empty string")

    try:
        response = create_gemini_client().models.embed_content(
            model=model_name,
            contents=f"task: search result | query: {text}",
        )
    except (APIError, GeminiConfigurationError) as error:
        raise GeminiEmbeddingError(
            f"Gemini could not generate a query embedding: {error}"
        ) from error

    if not response.embeddings or response.embeddings[0].values is None:
        raise GeminiEmbeddingError("Gemini returned no query embedding.")
    return [float(value) for value in response.embeddings[0].values]
