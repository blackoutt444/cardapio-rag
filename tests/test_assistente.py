import json

import pytest

from cardapio_rag.assistente import Assistente, RespostaInvalidaError
from cardapio_rag.llm import Geracao, ProvedorFake


def _json(encontrado=True, resposta="ok", itens_ids=()):
    return json.dumps({"encontrado": encontrado, "resposta": resposta, "itens_ids": list(itens_ids)})


class LLMComUso:
    """Provedor de teste que informa tokens e tempos, como o Ollama faz."""

    nome = modelo = "stub"

    def __init__(self, textos):
        self.textos = list(textos)

    def gerar_json(self, sistema, usuario, schema):
        return Geracao(
            texto=self.textos.pop(0),
            tokens_entrada=100,
            tokens_saida=10,
            carga_s=0.0,
            prompt_eval_s=1.0,
            geracao_s=0.5,
        )


def test_tempos_por_etapa(assistente):
    tempos = assistente.perguntar("pirarucu").tempos_ms
    assert set(tempos) == {"embedding", "busca", "prompt", "llm", "total"}
    assert all(v >= 0 for v in tempos.values())
    partes = tempos["embedding"] + tempos["busca"] + tempos["prompt"] + tempos["llm"]
    assert tempos["total"] >= partes - 0.01  # o total ainda inclui a checagem de fidelidade


def test_uso_do_llm_soma_as_tentativas(indice):
    llm = LLMComUso(["json quebrado", _json(itens_ids=["SR04"])])
    uso = Assistente(indice, llm).perguntar("pirarucu").uso_llm
    assert uso == {
        "tokens_entrada": 200,
        "tokens_saida": 20,
        "carga_s": 0.0,
        "prompt_eval_s": 2.0,
        "geracao_s": 1.0,
    }


def test_provedor_sem_metricas_deixa_uso_vazio(assistente):
    assert set(assistente.perguntar("pirarucu").uso_llm.values()) == {None}


def test_resposta_valida_cita_item_do_contexto(assistente, fake):
    resultado = assistente.perguntar("Tem algum prato com pirarucu?")

    assert resultado.resposta.encontrado is True
    assert resultado.resposta.itens_ids == ["SR04"]
    assert resultado.tentativas == 1
    assert resultado.ids_descartados == []
    # O contexto recuperado foi parar no prompt, com os ids.
    assert "[id=SR04]" in fake.chamadas[0]["usuario"]
    assert "PERGUNTA: Tem algum prato com pirarucu?" in fake.chamadas[0]["usuario"]


def test_schema_da_resposta_vai_para_o_provedor(assistente, fake):
    assistente.perguntar("pirarucu")
    schema = fake.chamadas[0]["schema"]
    assert set(schema["required"]) == {"encontrado", "resposta", "itens_ids"}
    assert schema["additionalProperties"] is False


def test_ids_fora_do_contexto_sao_descartados(indice):
    # ZZ99 não existe; SR04 está no contexto da pergunta.
    llm = ProvedorFake(respostas=[_json(itens_ids=["SR04", "ZZ99", "SR04"])])
    resultado = Assistente(indice, llm).perguntar("pirarucu")

    assert resultado.resposta.itens_ids == ["SR04"]
    assert resultado.ids_descartados == ["ZZ99"]


def test_json_invalido_gera_nova_tentativa_com_o_erro(indice):
    llm = ProvedorFake(respostas=["isto não é json", _json(itens_ids=["SR04"])])
    resultado = Assistente(indice, llm).perguntar("pirarucu")

    assert resultado.tentativas == 2
    assert "rejeitada pela validação" in llm.chamadas[1]["usuario"]


def test_campo_faltando_tambem_e_invalido(indice):
    sem_resposta = json.dumps({"encontrado": True, "itens_ids": []})
    llm = ProvedorFake(respostas=[sem_resposta, sem_resposta])
    with pytest.raises(RespostaInvalidaError):
        Assistente(indice, llm).perguntar("pirarucu")
    assert len(llm.chamadas) == 2


def test_k_controla_tamanho_do_contexto(assistente):
    assert len(assistente.perguntar("sobremesa", k=3).contexto) == 3
