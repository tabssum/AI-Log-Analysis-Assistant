from functools import lru_cache
import os

from google import genai
from streamlit.errors import StreamlitSecretNotFoundError
import streamlit as st


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini is not configured with an API key."""


def _get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
        except StreamlitSecretNotFoundError:
            api_key = ""

    if not api_key:
        raise GeminiConfigurationError(
            "Gemini is not configured. Set GOOGLE_API_KEY as an environment "
            "variable or add it to Streamlit secrets."
        )
    return api_key


@lru_cache(maxsize=1)
def get_gemini_client(api_key: str) -> genai.Client:
    """Return a cached Gemini API client for the configured key."""
    return genai.Client(api_key=api_key)


def create_gemini_client() -> genai.Client:
    return get_gemini_client(_get_api_key())
