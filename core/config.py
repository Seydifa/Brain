"""
config.py — Central backend configuration for the Brain system.

Controls which LLM backend is used (Gemini API or local Ollama), and
assigns **per-role models** so different agents can run on different LLMs.

Roles
-----
  purpose  — general reasoning (direction, memory, orchestrator, QA, …)
  coder    — code execution + tool calling (action agent, search agent)

Any role name is accepted; unrecognised roles fall back to the default model.

Usage
-----
  from core.config import configure, get_llm, get_embeddings, CFG

  # Quick setup — llama for reasoning, coder model for tool agents
  configure(
      backend='ollama',
      roles={
          'purpose': 'llama3.1:8b',
          'coder':   'qwen2.5-coder:7b',
      },
  )

  llm = get_llm(role='coder', temperature=0)

Environment variables (used as defaults, overridden by configure()):
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
# Role → agent mapping (informational, used by describe())
# ---------------------------------------------------------------------------

ROLE_AGENTS: dict[str, list[str]] = {
    "purpose": [
        "direction",
        "memory_classify",
        "memory_store",
        "search_validator",
        "orchestrator",
        "qa",
        "goal_evaluator",
    ],
    "coder": ["action", "search"],
}


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

    # Role → model overrides.  None = use the default model for the backend.
    roles: dict[str, str | None] = field(
        default_factory=lambda: {"purpose": None, "coder": None}
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
            # Auto-size max_loaded to the number of unique models
            n_unique = len(self.unique_models())
            if n_unique > self.ollama_max_loaded_models:
                self.ollama_max_loaded_models = n_unique
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(self.ollama_max_loaded_models)

    # ---- Role resolution ---------------------------------------------------

    @property
    def _default_model(self) -> str:
        return self.gemini_model if self.backend == "gemini" else self.ollama_model

    def model_for_role(self, role: str) -> str:
        """Return the model name assigned to *role*, falling back to default."""
        override = self.roles.get(role)
        return override if override else self._default_model

    def unique_models(self) -> set[str]:
        """Return the set of unique model names across all configured roles."""
        models = {self._default_model}
        for m in self.roles.values():
            if m:
                models.add(m)
        return models


# Singleton — import this everywhere
CFG = BackendConfig()


# ---------------------------------------------------------------------------
# configure() — the one-call setup
# ---------------------------------------------------------------------------


def configure(
    backend: str | None = None,
    roles: dict[str, str] | None = None,
    *,
    gemini_model: str | None = None,
    gemini_api_key: str | None = None,
    ollama_model: str | None = None,
    ollama_embed_model: str | None = None,
    ollama_host: str | None = None,
    ollama_max_loaded_models: int | None = None,
) -> BackendConfig:
    """
    Configure the Brain backend and per-role model assignments.

    Call this **before** importing the graph or any agents.

    Parameters
    ----------
    backend : str, optional
        'gemini' or 'ollama'.
    roles : dict, optional
        Maps role names to model names.
        Known roles: 'purpose' (general reasoning), 'coder' (tool agents).
        Unknown roles are accepted and resolved by get_llm(role=...).
    gemini_model / ollama_model : str, optional
        Override the default model for the backend.
    ollama_host : str, optional
        Override the Ollama server URL.
    ollama_max_loaded_models : int, optional
        Max models kept hot.  Auto-raised to len(unique_models) if needed.

    Returns
    -------
    BackendConfig
        The updated global CFG singleton.
    """
    global CFG

    if backend is not None:
        CFG.backend = backend
    if gemini_model is not None:
        CFG.gemini_model = gemini_model
    if gemini_api_key is not None:
        CFG.gemini_api_key = gemini_api_key
    if ollama_model is not None:
        CFG.ollama_model = ollama_model
    if ollama_embed_model is not None:
        CFG.ollama_embed_model = ollama_embed_model
    if ollama_host is not None:
        CFG.ollama_host = ollama_host
    if ollama_max_loaded_models is not None:
        CFG.ollama_max_loaded_models = ollama_max_loaded_models
    if roles:
        CFG.roles.update(roles)

    # Re-validate (also auto-sizes ollama_max_loaded_models)
    CFG.__post_init__()
    return CFG


# ---------------------------------------------------------------------------
# LLM / Embeddings factory
# ---------------------------------------------------------------------------


def get_llm(temperature: float = 0, role: str = "purpose"):
    """Return a chat LLM for the given role, configured for the active backend."""
    model = CFG.model_for_role(role)

    if CFG.backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=CFG.gemini_api_key,
        )
    else:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
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
    """Return a multi-line diagnostic summary of the active configuration."""
    lines = [f"backend : {CFG.backend}"]

    if CFG.backend == "gemini":
        key_hint = f"...{CFG.gemini_api_key[-6:]}" if CFG.gemini_api_key else "NOT SET"
        lines.append(f"api_key : {key_hint}")
    else:
        lines.append(f"host    : {CFG.ollama_host}")
        lines.append(f"embed   : {CFG.ollama_embed_model}")

    lines.append(f"default : {CFG._default_model}")
    lines.append("")
    lines.append("Role assignments:")
    for role_name in sorted(CFG.roles.keys()):
        model = CFG.model_for_role(role_name)
        agents = ROLE_AGENTS.get(role_name, [])
        agents_str = ", ".join(agents) if agents else "(custom)"
        override = " (default)" if model == CFG._default_model else ""
        lines.append(f"  {role_name:<10} → {model}{override}  [{agents_str}]")

    unique = CFG.unique_models()
    lines.append("")
    lines.append(f"Unique models: {len(unique)}  {sorted(unique)}")
    if CFG.backend == "ollama":
        lines.append(f"OLLAMA_MAX_LOADED_MODELS: {CFG.ollama_max_loaded_models}")

    return "\n".join(lines)
