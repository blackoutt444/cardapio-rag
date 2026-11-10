"""Testes de integridade dos dados: um erro aqui invalidaria a avaliação."""

import json

from cardapio_rag.config import RAIZ


def test_cardapio_tem_ids_unicos_e_precos_validos(itens):
    ids = [item.id for item in itens]
    assert len(ids) == len(set(ids))
    assert all(item.preco > 0 for item in itens)


def test_tags_estao_no_vocabulario(itens):
    permitidas = set(
        json.loads((RAIZ / "dados" / "cardapio.json").read_text(encoding="utf-8"))["tags_permitidas"]
    )
    for item in itens:
        assert set(item.tags) <= permitidas, item.id


def test_perguntas_de_referencia_apontam_para_itens_existentes(itens):
    ids = {item.id for item in itens}
    perguntas = json.loads((RAIZ / "avaliacao" / "perguntas.json").read_text(encoding="utf-8"))["perguntas"]
    assert len({p["id"] for p in perguntas}) == len(perguntas)
    for p in perguntas:
        assert set(p["esperados"]) <= ids, p["id"]
