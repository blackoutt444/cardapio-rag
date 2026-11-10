import pytest
from fastapi.testclient import TestClient

from cardapio_rag.api import criar_app
from cardapio_rag.assistente import Assistente
from cardapio_rag.llm import ErroProvedorLLM, ProvedorFake


@pytest.fixture
def cliente(assistente):
    # "with" executa o lifespan; como passamos o Assistente, nada é baixado.
    with TestClient(criar_app(assistente)) as c:
        yield c


def test_saude(cliente):
    corpo = cliente.get("/saude").json()
    assert corpo["status"] == "ok"
    assert corpo["itens_indexados"] > 0
    assert corpo["provedor"] == "fake"


def test_buscar(cliente):
    resposta = cliente.get("/buscar", params={"q": "pirarucu", "k": 3})
    assert resposta.status_code == 200
    resultados = resposta.json()["resultados"]
    assert len(resultados) == 3
    assert resultados[0]["item"]["id"] == "SR04"


def test_buscar_valida_parametros(cliente):
    assert cliente.get("/buscar", params={"q": "a"}).status_code == 422
    assert cliente.get("/buscar", params={"q": "pizza", "k": 0}).status_code == 422


def test_perguntar(cliente):
    resposta = cliente.post("/perguntar", json={"pergunta": "Tem algum prato com pirarucu?", "k": 3})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["encontrado"] is True
    assert [i["id"] for i in corpo["itens_citados"]] == ["SR04"]
    assert len(corpo["contexto"]) == 3
    assert corpo["provedor"] == "fake"
    assert set(corpo["tempos_ms"]) == {"embedding", "busca", "prompt", "llm", "total"}
    assert "tokens_entrada" in corpo["uso_llm"]


@pytest.mark.parametrize(
    "corpo", [{"pergunta": ""}, {"pergunta": "x" * 301}, {"pergunta": "pizza", "k": 50}, {}]
)
def test_perguntar_valida_entrada(cliente, corpo):
    assert cliente.post("/perguntar", json=corpo).status_code == 422


def test_provedor_fora_do_ar_retorna_503(indice):
    llm = ProvedorFake(erro=ErroProvedorLLM("conexão recusada"))
    with TestClient(criar_app(Assistente(indice, llm))) as c:
        resposta = c.post("/perguntar", json={"pergunta": "pirarucu"})
    assert resposta.status_code == 503


def test_json_invalido_retorna_502(indice):
    llm = ProvedorFake(respostas=["{", "{"])
    with TestClient(criar_app(Assistente(indice, llm))) as c:
        resposta = c.post("/perguntar", json={"pergunta": "pirarucu"})
    assert resposta.status_code == 502
