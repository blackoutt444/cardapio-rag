"""Modelos Pydantic: o contrato da resposta do LLM e o da API HTTP."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .dados import ItemCardapio


class RespostaLLM(BaseModel):
    """Formato que o LLM é obrigado a seguir.

    O JSON Schema deste modelo vai para o provedor (structured output) e, quando a
    resposta volta, o mesmo modelo valida o JSON. extra="forbid" gera
    "additionalProperties": false, exigido pelo modo estrito da OpenAI.
    As descrições dos campos entram no schema e funcionam como instrução para o modelo.
    """

    model_config = ConfigDict(extra="forbid")

    encontrado: bool = Field(
        description="true se os itens do CONTEXTO respondem a pergunta; false caso contrário."
    )
    resposta: str = Field(description="Resposta curta em português do Brasil para o cliente.")
    itens_ids: list[str] = Field(
        description='IDs (ex.: "SR01") dos itens do CONTEXTO usados na resposta. Vazio se encontrado=false.'
    )


# ---------- API ----------


class PerguntaRequest(BaseModel):
    pergunta: str = Field(min_length=3, max_length=300, examples=["Tem algum hambúrguer vegano?"])
    k: int = Field(default=5, ge=1, le=10, description="Quantos itens recuperar como contexto.")


class ItemRecuperado(BaseModel):
    item: ItemCardapio
    score: float


class PerguntaResponse(BaseModel):
    encontrado: bool
    resposta: str
    itens_citados: list[ItemCardapio]
    contexto: list[ItemRecuperado]
    ids_descartados: list[str] = Field(
        description="IDs citados pelo LLM que não estavam no contexto (possível alucinação)."
    )
    provedor: str
    modelo: str
    tentativas: int
    latencia_ms: int
    tempos_ms: dict[str, float] = Field(
        description="Tempo por etapa: embedding, busca, prompt, llm (inclui validação) e total."
    )
    uso_llm: dict[str, float | None] = Field(
        description="Tokens e tempos informados pelo provedor (o Ollama separa carga, prompt e geração)."
    )


class BuscaResponse(BaseModel):
    consulta: str
    resultados: list[ItemRecuperado]
