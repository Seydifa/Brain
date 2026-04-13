"""
config.py — Central backend configuration for the Brain system.

Controls whether the system uses Gemini API or a local Ollama server.
All agents import their LLM/embedding instances from here — never
directly from langchain_google_genai.

Usage
-----
  from config import get_llm, get_embeddings, CFG

Environment variables (take precedence over defaults):
  BRAIN_BACKEND         'gemini' | 'ollama'            (default: 'gemini')
  BRAIN_GEMINI_MODEL    e.g. 'gemini-2.5-pro'          (default below)
  BRAIN_OLLAMA_MODEL    e.g. 'llama3.1:8b'             (default below)
  BRAIN_OLLAMA_HOST     e.g. 'http://localhost:11434'  (default below)
  BRAIN_OLLAMA_MAX_LOADED_MODELS  '2'                  (default below)
  GOOGLE_API_KEY / GEMINI_API_KEY                      required for gemini
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    # Which backend to use
    backend: str = field(
        default_factory=lambda: os.environ.get("BRAIN_BACKEND", "gemini")
    )

    # Gemini settings
    gemini_model: str = field(
        default_factory=lambda: os.environ.get("BRAIN_GEMINI_MODEL", "gemini-2.5-pro")
    )
    gemini_embed_model: str = "models/gemini-embedding-001"
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        )
    )

    # Ollama settings
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("BRAIN_OLLAMA_MODEL", "llama3.1:8b")
    )
    ollama_embed_model: str = "nomic-embed-text"
    ollama_host: str = field(
        default_factory=lambda: os.environ.get(
            "BRAIN_OLLAMA_HOST", "http://localhost:11434"
        )
    )
    ollama_max_loaded_models: int = field(
        default_factory=lambda: int(
            os.environ.get("BRAIN_OLLAMA_MAX_LOADED_MODELS", "2")
        )
    )

    def __post_init__(self) -> None:
        if self.backend not in ("gemini", "ollama"):
            raise ValueError(
                f"BRAIN_BACKEND must be 'gemini' or 'ollama', got: {self.backend!r}"
            )
        if self.backend == "gemini" and not self.gemini_api_key:
            raise EnvironmentError(
                "BRAIN_BACKEND=gemini but GOOGLE_API_KEY is not set. "
                "Add it to .env or set GOOGLE_API_KEY in the environment."
            )
        if self.backend == "ollama":
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(self.ollama_max_loaded_models)


# Singleton — import this everywhere
CFG = BackendConfig()


# ---------------------------------------------------------------------------
# LLM / Embeddings factory
# ---------------------------------------------------------------------------


def get_llm(temperature: float = 0):
    """Return a chat LLM configured for the active backend."""
    if CFG.backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=CFG.gemini_model,
            temperature=temperature,
            google_api_key=CFG.gemini_api_key,
        )
    else:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=CFG.ollama_model,
            temperature=temperature,
            base_url=CFG.ollama_host,
        )


def get_embeddings():
    """Return an embeddings model configured for the active backend."""
    if CFG.backend == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=CFG.gemini_embed_model,
            google_api_key=CFG.gemini_api_key,
        )
    else:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=CFG.ollama_embed_model,
            base_url=CFG.ollama_host,
        )


def describe() -> str:
    """Return a one-line summary of the active backend for diagnostics."""
    if CFG.backend == "gemini":
        key_hint = f"...{CFG.gemini_api_key[-6:]}" if CFG.gemini_api_key else "NOT SET"
        return f"backend=gemini  model={CFG.gemini_model}  key={key_hint}"
    return (
        f"backend=ollama  model={CFG.ollama_model}  "
        f"embed={CFG.ollama_embed_model}  "
        f"host={CFG.ollama_host}  "
        f"max_loaded={CFG.ollama_max_loaded_models}"
    )
