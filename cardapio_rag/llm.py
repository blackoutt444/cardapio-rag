"""Provedores de LLM atrás de uma interface única.

O resto do código só conhece `ProvedorLLM.gerar_json(...)`. Trocar Ollama por OpenAI
ou Anthropic é mudar uma variável de ambiente, e os testes usam o `ProvedorFake`.

Todos os provedores devolvem uma `Geracao`: o JSON como texto cru, mais tokens e
tempos quando o provedor informa. A validação fica num lugar só (o Assistente),
assim a avaliação consegue medir quantas respostas vieram inválidas.

As chamadas HTTP usam httpx direto, sem os SDKs oficiais: são poucas linhas, não há
dependência extra e os testes simulam as APIs com httpx.MockTransport.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import Config

MODELOS_PADRAO = {
    "ollama": "qwen2.5:3b",
    "openai": "gpt-5-mini",
    "anthropic": "claude-haiku-5-5",
    "fake": "fake",
}


class ErroProvedorLLM(RuntimeError):
    """O provedor não respondeu (fora do ar, timeout, chave inválida...)."""


@dataclass(frozen=True)
class Geracao:
    """O texto gerado e o que o provedor informar sobre custo e tempo da chamada."""

    texto: str
    tokens_entrada: int | None = None
    tokens_saida: int | None = None
    # Só o Ollama decompõe o tempo de inferência (em segundos):
    carga_s: float | None = None  # carregar o modelo na memória (alto na 1ª chamada)
    prompt_eval_s: float | None = None  # processar os tokens de entrada
    geracao_s: float | None = None  # gerar os tokens de saída, um a um


class ProvedorLLM(Protocol):
    nome: str
    modelo: str

    def gerar_json(self, sistema: str, usuario: str, schema: dict[str, Any]) -> Geracao: ...


def _post_json(
    cliente: httpx.Client, url: str, corpo: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    try:
        resposta = cliente.post(url, json=corpo, headers=headers)
    except httpx.HTTPError as erro:
        raise ErroProvedorLLM(f"falha ao chamar {url}: {erro}") from erro
    if resposta.status_code >= 400:
        raise ErroProvedorLLM(f"{url} respondeu {resposta.status_code}: {resposta.text[:300]}")
    return resposta.json()


class ProvedorOllama:
    """LLM local via Ollama. O campo "format" recebe o JSON Schema e o Ollama restringe
    a geração para que a saída siga o schema (decodificação restrita)."""

    nome = "ollama"

    def __init__(
        self,
        modelo: str = MODELOS_PADRAO["ollama"],
        url: str = "http://localhost:11434",
        timeout_s: float = 180.0,
        cliente: httpx.Client | None = None,
    ) -> None:
        self.modelo = modelo
        self.url = url.rstrip("/")
        self.cliente = cliente or httpx.Client(timeout=timeout_s)

    def gerar_json(self, sistema: str, usuario: str, schema: dict[str, Any]) -> Geracao:
        corpo = {
            "model": self.modelo,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": sistema},
                {"role": "user", "content": usuario},
            ],
        }
        dados = _post_json(self.cliente, f"{self.url}/api/chat", corpo)
        # O Ollama informa as durações em nanossegundos.
        return Geracao(
            texto=dados["message"]["content"],
            tokens_entrada=dados.get("prompt_eval_count"),
            tokens_saida=dados.get("eval_count"),
            carga_s=dados.get("load_duration", 0) / 1e9,
            prompt_eval_s=dados.get("prompt_eval_duration", 0) / 1e9,
            geracao_s=dados.get("eval_duration", 0) / 1e9,
        )


class ProvedorOpenAI:
    """Chat Completions com response_format json_schema em modo estrito."""

    nome = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        chave: str,
        modelo: str = MODELOS_PADRAO["openai"],
        timeout_s: float = 60.0,
        cliente: httpx.Client | None = None,
    ) -> None:
        self.modelo = modelo
        self.chave = chave
        self.cliente = cliente or httpx.Client(timeout=timeout_s)

    def gerar_json(self, sistema: str, usuario: str, schema: dict[str, Any]) -> Geracao:
        corpo = {
            "model": self.modelo,
            "messages": [
                {"role": "system", "content": sistema},
                {"role": "user", "content": usuario},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "resposta_cardapio", "schema": schema, "strict": True},
            },
        }
        headers = {"Authorization": f"Bearer {self.chave}"}
        dados = _post_json(self.cliente, self.URL, corpo, headers)
        uso = dados.get("usage", {})
        return Geracao(
            texto=dados["choices"][0]["message"]["content"],
            tokens_entrada=uso.get("prompt_tokens"),
            tokens_saida=uso.get("completion_tokens"),
        )


class ProvedorAnthropic:
    """Messages API. A saída estruturada vem de uma ferramenta obrigatória
    (tool_choice) cujo input_schema é o schema da resposta: o modelo "chama" a
    ferramenta e o argumento dela é o nosso JSON."""

    nome = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        chave: str,
        modelo: str = MODELOS_PADRAO["anthropic"],
        timeout_s: float = 60.0,
        cliente: httpx.Client | None = None,
    ) -> None:
        self.modelo = modelo
        self.chave = chave
        self.cliente = cliente or httpx.Client(timeout=timeout_s)

    def gerar_json(self, sistema: str, usuario: str, schema: dict[str, Any]) -> Geracao:
        corpo = {
            "model": self.modelo,
            "max_tokens": 1024,
            "system": sistema,
            "messages": [{"role": "user", "content": usuario}],
            "tools": [
                {
                    "name": "responder",
                    "description": "Registra a resposta final para o cliente.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "responder"},
        }
        headers = {"x-api-key": self.chave, "anthropic-version": "2023-06-01"}
        dados = _post_json(self.cliente, self.URL, corpo, headers)
        uso = dados.get("usage", {})
        for bloco in dados.get("content", []):
            if bloco.get("type") == "tool_use":
                return Geracao(
                    texto=json.dumps(bloco["input"], ensure_ascii=False),
                    tokens_entrada=uso.get("input_tokens"),
                    tokens_saida=uso.get("output_tokens"),
                )
        raise ErroProvedorLLM("a resposta da Anthropic não trouxe o bloco tool_use")


class ProvedorFake:
    """LLM falso e determinístico, usado nos testes e no CI.

    Sem `respostas`: cita o primeiro item do contexto, ou responde "não encontrado"
    se o contexto estiver vazio. Com `respostas`: devolve cada uma em ordem, o que
    permite simular JSON inválido. `erro` simula o provedor fora do ar.
    """

    nome = "fake"
    modelo = "fake"

    def __init__(self, respostas: Iterable[str] | None = None, erro: Exception | None = None) -> None:
        self._respostas = list(respostas) if respostas is not None else None
        self._erro = erro
        self.chamadas: list[dict[str, Any]] = []

    def gerar_json(self, sistema: str, usuario: str, schema: dict[str, Any]) -> Geracao:
        self.chamadas.append({"sistema": sistema, "usuario": usuario, "schema": schema})
        if self._erro is not None:
            raise self._erro
        if self._respostas is not None:
            return Geracao(texto=self._respostas.pop(0))
        ids = re.findall(r"\[id=(\w+)\]", usuario)
        if not ids:
            texto = json.dumps({"encontrado": False, "resposta": "Não encontrei.", "itens_ids": []})
        else:
            texto = json.dumps(
                {"encontrado": True, "resposta": f"Sugiro o item {ids[0]}.", "itens_ids": [ids[0]]}
            )
        return Geracao(texto=texto)


def criar_provedor(config: Config) -> ProvedorLLM:
    nome = config.llm_provedor
    if nome not in MODELOS_PADRAO:
        raise ValueError(f"LLM_PROVEDOR inválido: {nome!r} (use {', '.join(MODELOS_PADRAO)})")
    modelo = config.llm_modelo or MODELOS_PADRAO[nome]

    if nome == "ollama":
        return ProvedorOllama(modelo, config.ollama_url, config.llm_timeout_s)
    if nome == "fake":
        return ProvedorFake()

    variavel = "OPENAI_API_KEY" if nome == "openai" else "ANTHROPIC_API_KEY"
    chave = os.getenv(variavel)
    if not chave:
        raise ValueError(f"LLM_PROVEDOR={nome} exige a variável {variavel}")
    if nome == "openai":
        return ProvedorOpenAI(chave, modelo)
    return ProvedorAnthropic(chave, modelo)
