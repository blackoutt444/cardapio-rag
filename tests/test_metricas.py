import pytest

from cardapio_rag.metricas import hit_at_k, media, precisao_citacoes, precos_no_texto, reciprocal_rank


@pytest.mark.parametrize(
    ("recuperados", "esperados", "k", "esperado"),
    [
        (["A", "B", "C"], {"A"}, 1, 1.0),  # acerto na 1ª posição
        (["A", "B", "C"], {"B"}, 1, 0.0),  # limite: k=1 só olha o primeiro
        (["A", "B", "C"], {"C"}, 2, 0.0),  # esperado logo depois do corte
        (["A", "B", "C"], {"C"}, 3, 1.0),  # limite: k igual ao tamanho da lista
        (["A", "B"], {"B"}, 10, 1.0),  # k maior que a lista não quebra
        (["A", "B"], {"C"}, 10, 0.0),  # nenhum relevante
        (["A", "B", "C"], {"Z", "B"}, 3, 1.0),  # basta um dos esperados
        (["A", "B", "C"], {"A", "B", "C"}, 3, 1.0),  # várias correspondências contam como 1
        ([], {"A"}, 5, 0.0),  # busca sem resultados
        (["A", "B"], set(), 2, 0.0),  # nada é relevante
        (["A", "A", "B"], {"B"}, 2, 0.0),  # id repetido ocupa posição
    ],
)
def test_hit_at_k(recuperados, esperados, k, esperado):
    assert hit_at_k(recuperados, esperados, k) == esperado


@pytest.mark.parametrize("k", [0, -1])
def test_hit_at_k_rejeita_k_invalido(k):
    with pytest.raises(ValueError, match="k deve ser >= 1"):
        hit_at_k(["A"], {"A"}, k)


def test_hit_at_k_e_monotonico_em_k():
    # Aumentar k nunca pode piorar o hit@k.
    recuperados = ["A", "B", "C", "D", "E"]
    valores = [hit_at_k(recuperados, {"D"}, k) for k in range(1, 6)]
    assert valores == [0.0, 0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("recuperados", "esperados", "esperado"),
    [
        (["A", "B", "C"], {"A"}, 1.0),
        (["A", "B", "C"], {"B", "C"}, 0.5),  # várias correspondências: conta só a primeira
        (["A", "B", "C", "D"], {"D"}, 0.25),
        (["A", "B", "C"], {"C", "A"}, 1.0),  # a ordem do conjunto não importa
        (["A", "B"], {"Z"}, 0.0),  # nenhum relevante
        ([], {"A"}, 0.0),  # busca sem resultados
        (["A", "B"], set(), 0.0),  # nada é relevante
    ],
)
def test_reciprocal_rank(recuperados, esperados, esperado):
    assert reciprocal_rank(recuperados, esperados) == pytest.approx(esperado)


def test_mrr_com_exemplo_conhecido():
    # Três perguntas: acerto na 1ª, na 3ª e nenhum acerto. MRR = (1 + 1/3 + 0) / 3 = 4/9.
    rrs = [
        reciprocal_rank(["A", "B"], {"A"}),
        reciprocal_rank(["X", "Y", "B"], {"B"}),
        reciprocal_rank(["X", "Y"], {"B"}),
    ]
    assert media(rrs) == pytest.approx(4 / 9)


@pytest.mark.parametrize(
    ("citados", "esperados", "esperado"),
    [
        (["A"], {"A"}, 1.0),
        (["A", "B"], {"A"}, 0.5),  # citação extra irrelevante
        (["A", "B"], {"A", "B", "C"}, 1.0),  # não precisa citar todos os relevantes
        (["X"], {"A"}, 0.0),
        ([], {"A"}, 0.0),  # nada citado
    ],
)
def test_precisao_citacoes(citados, esperados, esperado):
    assert precisao_citacoes(citados, esperados) == pytest.approx(esperado)


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("A lasanha custa R$ 58,00.", [58.0]),
        ("Sai por R$58", [58.0]),
        ("R$ 64,9 no Sabor do Rio", [64.9]),
        ("Total de R$ 1.234,50", [1234.5]),
        ("Custa R$ 22.90", [22.9]),  # ponto como separador decimal
        ("Burger R$ 36,90 e batata R$ 19,90.", [36.9, 19.9]),
        ("Não encontrei esse prato.", []),
        ("São 3 opções", []),  # número sem R$ não é preço
    ],
)
def test_precos_no_texto(texto, esperado):
    assert precos_no_texto(texto) == pytest.approx(esperado)


def test_media():
    assert media([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert media([]) == 0.0
