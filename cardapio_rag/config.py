"""Configuração lida de variáveis de ambiente, com padrões que rodam 100% local."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # Qual LLM gera a resposta: "ollama" (padrão, local), "openai", "anthropic" ou "fake".
    llm_provedor: str = "ollama"
    # None = usa o modelo padrão do provedor (veja llm.MODELOS_PADRAO).
    llm_modelo: str | None = None
    ollama_url: str = "http://localhost:11434"
    llm_timeout_s: float = 180.0
    # Como os textos viram vetores: "e5" (semântico, local) ou "hash" (lexical, sem dependências).
    embeddings: str = "e5"
    embeddings_modelo: str = "intfloat/multilingual-e5-small"
    cardapio_path: Path = RAIZ / "dados" / "cardapio.json"
    top_k: int = 5

    @classmethod
    def do_ambiente(cls) -> Config:
        return cls(
            llm_provedor=os.getenv("LLM_PROVEDOR", cls.llm_provedor).lower(),
            llm_modelo=os.getenv("LLM_MODELO") or None,
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", cls.llm_timeout_s)),
            embeddings=os.getenv("EMBEDDINGS", cls.embeddings).lower(),
            embeddings_modelo=os.getenv("EMBEDDINGS_MODELO", cls.embeddings_modelo),
            cardapio_path=Path(os.getenv("CARDAPIO_PATH", str(cls.cardapio_path))),
            top_k=int(os.getenv("TOP_K", cls.top_k)),
        )
