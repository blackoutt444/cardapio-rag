"""Carrega o cardápio e define como cada item vira texto para a busca."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ItemCardapio(BaseModel):
    id: str
    restaurante: str
    nome: str
    descricao: str
    categoria: str
    preco: float = Field(gt=0)
    tags: list[str] = Field(default_factory=list)

    def texto_para_busca(self) -> str:
        """Texto que vira embedding. Cada item é um "chunk": o dado já é estruturado e
        curto, então não faz sentido quebrar em pedaços menores. O preço fica de fora
        porque número não ajuda na similaridade semântica (é tratado por filtro)."""
        tags = ", ".join(self.tags) if self.tags else "nenhuma"
        return (
            f"{self.nome}. {self.descricao} Categoria: {self.categoria}. "
            f"Restaurante: {self.restaurante}. Tags: {tags}."
        )


def carregar_cardapio(caminho: Path) -> list[ItemCardapio]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    itens = [ItemCardapio.model_validate(item) for item in dados["itens"]]
    ids = [item.id for item in itens]
    repetidos = {i for i in ids if ids.count(i) > 1}
    if repetidos:
        raise ValueError(f"ids repetidos no cardápio: {sorted(repetidos)}")
    return itens
