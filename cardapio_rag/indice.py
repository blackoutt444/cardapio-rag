"""Índice vetorial em memória: busca exata por similaridade de cosseno com numpy.

Com ~50 itens, comparar a pergunta com todos os itens custa microssegundos. Um banco
vetorial (FAISS, pgvector, Qdrant...) só passa a valer a pena com dezenas de
milhares de documentos ou quando o índice precisa ser persistido e compartilhado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dados import ItemCardapio
from .embeddings import Embedder, tokenizar


@dataclass(frozen=True)
class ResultadoBusca:
    item: ItemCardapio
    score: float


class IndiceVetorial:
    def __init__(self, itens: list[ItemCardapio], embedder: Embedder) -> None:
        self.itens = itens
        self.embedder = embedder
        # Calculado uma vez na inicialização: matriz (n_itens, dimensão).
        self.matriz: np.ndarray = embedder.embed_documentos([i.texto_para_busca() for i in itens])

    def buscar(
        self,
        consulta: str,
        k: int = 5,
        preco_max: float | None = None,
        tag: str | None = None,
    ) -> list[ResultadoBusca]:
        return self.buscar_por_vetor(self.embedder.embed_consulta(consulta), k, preco_max, tag)

    def buscar_por_vetor(
        self,
        vetor_consulta: np.ndarray,
        k: int = 5,
        preco_max: float | None = None,
        tag: str | None = None,
    ) -> list[ResultadoBusca]:
        """Separado de `buscar` para medir o embedding da pergunta e a busca em si."""
        # Vetores normalizados: produto escalar = similaridade de cosseno.
        scores = self.matriz @ vetor_consulta

        # Filtros de metadados antes do ranking (preço e tag não são semânticos).
        tag_normalizada = " ".join(tokenizar(tag)) if tag else None
        candidatos = [
            i
            for i, item in enumerate(self.itens)
            if (preco_max is None or item.preco <= preco_max)
            and (tag_normalizada is None or tag_normalizada in {" ".join(tokenizar(t)) for t in item.tags})
        ]
        candidatos.sort(key=lambda i: scores[i], reverse=True)
        return [ResultadoBusca(self.itens[i], float(scores[i])) for i in candidatos[:k]]
