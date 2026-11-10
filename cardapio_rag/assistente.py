"""O pipeline RAG: recuperar -> montar prompt -> gerar -> validar -> checar fidelidade."""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import ValidationError

from .config import Config
from .dados import carregar_cardapio
from .embeddings import criar_embedder
from .esquemas import RespostaLLM
from .indice import IndiceVetorial, ResultadoBusca
from .llm import Geracao, ProvedorLLM, criar_provedor
from .prompt import SISTEMA, montar_mensagem_usuario

SCHEMA_RESPOSTA = RespostaLLM.model_json_schema()
CAMPOS_USO = ("tokens_entrada", "tokens_saida", "carga_s", "prompt_eval_s", "geracao_s")


class RespostaInvalidaError(RuntimeError):
    """O LLM não devolveu um JSON válido mesmo depois de nova tentativa."""


@dataclass(frozen=True)
class Resultado:
    resposta: RespostaLLM
    contexto: list[ResultadoBusca]
    ids_descartados: list[str]
    tentativas: int
    # Tempo de cada etapa em ms: embedding, busca, prompt, llm (com validação) e total.
    tempos_ms: dict[str, float]
    # Tokens e tempos informados pelo provedor, somados entre as tentativas.
    uso_llm: dict[str, float | None]

    @property
    def latencia_ms(self) -> int:
        return round(self.tempos_ms["total"])


class Assistente:
    def __init__(self, indice: IndiceVetorial, llm: ProvedorLLM, k: int = 5, max_tentativas: int = 2) -> None:
        self.indice = indice
        self.llm = llm
        self.k = k
        self.max_tentativas = max_tentativas

    def perguntar(self, pergunta: str, k: int | None = None) -> Resultado:
        marcas = [time.perf_counter()]

        # 1. Recuperação: embedding da pergunta e os k itens mais parecidos.
        vetor = self.indice.embedder.embed_consulta(pergunta)
        marcas.append(time.perf_counter())
        contexto = self.indice.buscar_por_vetor(vetor, k=k or self.k)
        marcas.append(time.perf_counter())

        # 2. Prompt com o contexto recuperado.
        mensagem = montar_mensagem_usuario(pergunta, contexto)
        marcas.append(time.perf_counter())

        # 3. Geração com saída estruturada + validação com Pydantic.
        resposta, tentativas, geracoes = self._gerar_validado(mensagem)
        marcas.append(time.perf_counter())

        # 4. Fidelidade: o LLM só pode citar itens que estavam no contexto. IDs fora
        # do contexto são alucinação, então saem da resposta e ficam registrados.
        ids_contexto = {r.item.id for r in contexto}
        citados = list(dict.fromkeys(resposta.itens_ids))  # remove repetidos, mantém a ordem
        descartados = [i for i in citados if i not in ids_contexto]
        resposta = resposta.model_copy(update={"itens_ids": [i for i in citados if i in ids_contexto]})
        marcas.append(time.perf_counter())

        etapas = ("embedding", "busca", "prompt", "llm")
        tempos_ms = {etapa: _ms(marcas[i], marcas[i + 1]) for i, etapa in enumerate(etapas)}
        tempos_ms["total"] = _ms(marcas[0], marcas[-1])
        return Resultado(resposta, contexto, descartados, tentativas, tempos_ms, _somar_uso(geracoes))

    def _gerar_validado(self, mensagem: str) -> tuple[RespostaLLM, int, list[Geracao]]:
        geracoes: list[Geracao] = []
        erro_anterior: str | None = None
        for tentativa in range(1, self.max_tentativas + 1):
            texto = mensagem
            if erro_anterior:
                # Nova tentativa mostrando ao modelo o que deu errado.
                texto += (
                    f"\n\nSua resposta anterior foi rejeitada pela validação: {erro_anterior}\n"
                    "Responda novamente apenas com o JSON no formato pedido."
                )
            geracao = self.llm.gerar_json(SISTEMA, texto, SCHEMA_RESPOSTA)
            geracoes.append(geracao)
            try:
                return RespostaLLM.model_validate_json(geracao.texto), tentativa, geracoes
            except ValidationError as erro:
                erro_anterior = _resumir_erro(erro)
        raise RespostaInvalidaError(f"JSON inválido após {self.max_tentativas} tentativas: {erro_anterior}")


def _ms(inicio: float, fim: float) -> float:
    return round((fim - inicio) * 1000, 3)


def _somar_uso(geracoes: list[Geracao]) -> dict[str, float | None]:
    uso: dict[str, float | None] = {}
    for campo in CAMPOS_USO:
        valores = [v for g in geracoes if (v := getattr(g, campo)) is not None]
        uso[campo] = sum(valores) if valores else None
    return uso


def _resumir_erro(erro: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'json'}: {e['msg']}" for e in erro.errors()[:3])


def criar_assistente(config: Config) -> Assistente:
    itens = carregar_cardapio(config.cardapio_path)
    indice = IndiceVetorial(itens, criar_embedder(config))
    return Assistente(indice, criar_provedor(config), k=config.top_k)
