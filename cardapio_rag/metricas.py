"""Métricas de recuperação usadas pela avaliação (avaliacao/avaliar.py).

`recuperados` é a lista de ids na ordem devolvida pela busca (o primeiro é o mais
parecido). `esperados` é o conjunto de ids que respondem a pergunta: basta um
deles aparecer para a busca ter acertado.
"""

from __future__ import annotations


def hit_at_k(recuperados: list[str], esperados: set[str], k: int) -> float:
    """1.0 se pelo menos um id esperado aparece entre os k primeiros recuperados, senão 0.0.

    Responde "o item certo chegou ao LLM?", já que só os k primeiros viram contexto.

    >>> hit_at_k(["A", "B", "C"], {"C"}, k=2)
    0.0
    >>> hit_at_k(["A", "B", "C"], {"C"}, k=3)
    1.0
    """
    if k < 1:
        raise ValueError(f"k deve ser >= 1, recebido {k}")
    # Fatiar além do fim da lista não dá erro: recuperados[:10] numa lista de 3 devolve as 3.
    return 1.0 if any(doc in esperados for doc in recuperados[:k]) else 0.0


def reciprocal_rank(recuperados: list[str], esperados: set[str]) -> float:
    """1 / posição do primeiro id esperado na lista (a posição começa em 1).
    Se nenhum esperado aparece, 0.0. A média disso sobre as perguntas é o MRR.

    Diferente do hit@k, premia o item certo aparecer mais no alto da lista.

    >>> reciprocal_rank(["A", "B", "C"], {"B", "C"})
    0.5
    >>> reciprocal_rank(["A", "B"], {"Z"})
    0.0
    """
    for posicao, doc in enumerate(recuperados, start=1):
        if doc in esperados:
            return 1.0 / posicao
    return 0.0


def media(valores: list[float]) -> float:
    """Média simples; 0.0 para lista vazia (evita divisão por zero)."""
    return sum(valores) / len(valores) if valores else 0.0
