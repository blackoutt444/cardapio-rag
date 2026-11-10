"""Transforma textos em vetores normalizados (norma 1).

Com vetores normalizados, o produto escalar entre dois vetores é igual à
similaridade de cosseno, então a busca vira uma multiplicação de matriz.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from typing import Protocol

import numpy as np

from .config import Config


class Embedder(Protocol):
    nome: str

    def embed_documentos(self, textos: list[str]) -> np.ndarray:
        """Retorna uma matriz (n_textos, dimensão) com uma linha por texto."""
        ...

    def embed_consulta(self, texto: str) -> np.ndarray:
        """Retorna um vetor (dimensão,) para a pergunta do usuário."""
        ...


class EmbedderE5:
    """Embeddings semânticos locais com sentence-transformers.

    O multilingual-e5-small (~118M parâmetros, 384 dimensões) entende português e
    roda em CPU. A família E5 foi treinada com prefixos: "query: " para perguntas e
    "passage: " para documentos. Sem os prefixos a qualidade da busca cai.
    """

    def __init__(self, modelo: str = "intfloat/multilingual-e5-small") -> None:
        # Import tardio: o PyTorch só é necessário quando este embedder é usado.
        # Assim os testes e o CI rodam sem instalar sentence-transformers.
        from sentence_transformers import SentenceTransformer

        self.nome = modelo
        self._modelo = SentenceTransformer(modelo, device="cpu")

    def embed_documentos(self, textos: list[str]) -> np.ndarray:
        return self._modelo.encode(
            [f"passage: {t}" for t in textos], normalize_embeddings=True, convert_to_numpy=True
        )

    def embed_consulta(self, texto: str) -> np.ndarray:
        return self._modelo.encode(f"query: {texto}", normalize_embeddings=True, convert_to_numpy=True)


STOPWORDS = set(
    "a o as os um uma uns umas de da do das dos e em no na nos nas com para por que "
    "tem algum alguma quero voces qual quais onde eu me se ou ao".split()
)


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sem acentos, só palavras, sem stopwords."""
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", sem_acento) if t not in STOPWORDS]


class EmbedderHash:
    """Baseline lexical: saco de palavras com feature hashing.

    Cada palavra cai numa posição do vetor (crc32 da palavra % dimensão). Não entende
    sinônimos, só palavras iguais. Serve de comparação na avaliação e roda nos testes
    e no CI sem baixar modelo nenhum. Usa crc32 (e não hash()) porque o hash() do
    Python muda a cada execução, o que deixaria os testes não determinísticos.
    """

    def __init__(self, dimensao: int = 2048) -> None:
        self.nome = f"hash-{dimensao}"
        self.dimensao = dimensao

    def _vetor(self, texto: str) -> np.ndarray:
        vetor = np.zeros(self.dimensao, dtype=np.float32)
        for token in tokenizar(texto):
            vetor[zlib.crc32(token.encode()) % self.dimensao] += 1.0
        norma = np.linalg.norm(vetor)
        return vetor / norma if norma > 0 else vetor

    def embed_documentos(self, textos: list[str]) -> np.ndarray:
        return np.stack([self._vetor(t) for t in textos])

    def embed_consulta(self, texto: str) -> np.ndarray:
        return self._vetor(texto)


def criar_embedder(config: Config) -> Embedder:
    if config.embeddings == "e5":
        return EmbedderE5(config.embeddings_modelo)
    if config.embeddings == "hash":
        return EmbedderHash()
    raise ValueError(f"EMBEDDINGS inválido: {config.embeddings!r} (use 'e5' ou 'hash')")
