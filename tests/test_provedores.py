"""Os provedores reais são testados sem rede: httpx.MockTransport intercepta a
requisição, deixa a gente conferir o que seria enviado e devolve uma resposta pronta."""

import json

import httpx
import pytest

from cardapio_rag.config import Config
from cardapio_rag.llm import (
    ErroProvedorLLM,
    ProvedorAnthropic,
    ProvedorOllama,
    ProvedorOpenAI,
    criar_provedor,
)

SCHEMA = {"type": "object", "properties": {"resposta": {"type": "string"}}}
RESPOSTA = '{"resposta": "oi"}'


def cliente_simulado(responder):
    """Cria um httpx.Client cujas requisições são respondidas por `responder` e
    guardadas em `cliente.requisicoes`."""
    requisicoes = []

    def handler(request: httpx.Request) -> httpx.Response:
        requisicoes.append(request)
        return responder(request)

    cliente = httpx.Client(transport=httpx.MockTransport(handler))
    cliente.requisicoes = requisicoes
    return cliente


def test_ollama_envia_schema_no_campo_format():
    cliente = cliente_simulado(lambda r: httpx.Response(200, json={"message": {"content": RESPOSTA}}))
    provedor = ProvedorOllama("qwen2.5:3b", "http://ollama:11434", cliente=cliente)

    assert provedor.gerar_json("sistema", "usuario", SCHEMA).texto == RESPOSTA

    requisicao = cliente.requisicoes[0]
    corpo = json.loads(requisicao.content)
    assert str(requisicao.url) == "http://ollama:11434/api/chat"
    assert corpo["format"] == SCHEMA
    assert corpo["stream"] is False
    assert corpo["options"]["temperature"] == 0
    assert [m["role"] for m in corpo["messages"]] == ["system", "user"]


def test_ollama_repassa_opcoes_e_aceita_schema_vazio():
    cliente = cliente_simulado(lambda r: httpx.Response(200, json={"message": {"content": RESPOSTA}}))
    ProvedorOllama(cliente=cliente, opcoes={"num_gpu": 0}).gerar_json("s", "u", {})

    corpo = json.loads(cliente.requisicoes[0].content)
    assert corpo["options"] == {"temperature": 0, "num_gpu": 0}
    assert "format" not in corpo  # sem schema: sem decodificação restrita


def test_fabrica_passa_num_gpu_para_o_ollama():
    provedor = criar_provedor(Config(llm_provedor="ollama", ollama_num_gpu=0))
    assert provedor.opcoes == {"num_gpu": 0}
    assert criar_provedor(Config(llm_provedor="ollama")).opcoes == {}


def test_ollama_converte_tempos_e_tokens():
    # Formato real do /api/chat: durações em nanossegundos.
    dados = {
        "message": {"content": RESPOSTA},
        "load_duration": 2_500_000_000,
        "prompt_eval_count": 600,
        "prompt_eval_duration": 12_000_000_000,
        "eval_count": 50,
        "eval_duration": 5_000_000_000,
    }
    cliente = cliente_simulado(lambda r: httpx.Response(200, json=dados))
    geracao = ProvedorOllama(cliente=cliente).gerar_json("s", "u", SCHEMA)

    assert (geracao.tokens_entrada, geracao.tokens_saida) == (600, 50)
    assert (geracao.carga_s, geracao.prompt_eval_s, geracao.geracao_s) == (2.5, 12.0, 5.0)


def test_openai_usa_json_schema_estrito():
    dados = {
        "choices": [{"message": {"content": RESPOSTA}}],
        "usage": {"prompt_tokens": 600, "completion_tokens": 40},
    }
    cliente = cliente_simulado(lambda r: httpx.Response(200, json=dados))
    provedor = ProvedorOpenAI("chave-de-teste", "modelo-x", cliente=cliente)

    geracao = provedor.gerar_json("sistema", "usuario", SCHEMA)
    assert geracao.texto == RESPOSTA
    assert (geracao.tokens_entrada, geracao.tokens_saida) == (600, 40)

    requisicao = cliente.requisicoes[0]
    corpo = json.loads(requisicao.content)
    assert requisicao.headers["Authorization"] == "Bearer chave-de-teste"
    assert corpo["response_format"]["json_schema"]["strict"] is True
    assert corpo["response_format"]["json_schema"]["schema"] == SCHEMA


def test_anthropic_extrai_json_do_tool_use():
    dados = {
        "content": [{"type": "tool_use", "name": "responder", "input": {"resposta": "oi"}}],
        "usage": {"input_tokens": 700, "output_tokens": 30},
    }
    cliente = cliente_simulado(lambda r: httpx.Response(200, json=dados))
    provedor = ProvedorAnthropic("chave-de-teste", "modelo-x", cliente=cliente)

    geracao = provedor.gerar_json("sistema", "usuario", SCHEMA)
    assert json.loads(geracao.texto) == {"resposta": "oi"}
    assert (geracao.tokens_entrada, geracao.tokens_saida) == (700, 30)

    requisicao = cliente.requisicoes[0]
    corpo = json.loads(requisicao.content)
    assert requisicao.headers["x-api-key"] == "chave-de-teste"
    assert corpo["tool_choice"] == {"type": "tool", "name": "responder"}
    assert corpo["tools"][0]["input_schema"] == SCHEMA
    assert corpo["system"] == "sistema"


def test_anthropic_sem_tool_use_e_erro():
    cliente = cliente_simulado(
        lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": "oi"}]})
    )
    with pytest.raises(ErroProvedorLLM):
        ProvedorAnthropic("k", cliente=cliente).gerar_json("s", "u", SCHEMA)


def test_status_de_erro_vira_erro_do_provedor():
    cliente = cliente_simulado(lambda r: httpx.Response(500, text="falhou"))
    with pytest.raises(ErroProvedorLLM, match="500"):
        ProvedorOllama(cliente=cliente).gerar_json("s", "u", SCHEMA)


def test_falha_de_conexao_vira_erro_do_provedor():
    def recusar(request):
        raise httpx.ConnectError("conexão recusada", request=request)

    with pytest.raises(ErroProvedorLLM, match="conexão recusada"):
        ProvedorOllama(cliente=cliente_simulado(recusar)).gerar_json("s", "u", SCHEMA)


def test_fabrica_exige_chave_para_provedor_pago(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        criar_provedor(Config(llm_provedor="openai"))


def test_fabrica_rejeita_provedor_desconhecido():
    with pytest.raises(ValueError, match="inválido"):
        criar_provedor(Config(llm_provedor="xyz"))
