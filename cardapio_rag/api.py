"""API HTTP (FastAPI). Rode com: uvicorn cardapio_rag.api:app --reload"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from .assistente import Assistente, RespostaInvalidaError, criar_assistente
from .config import Config
from .esquemas import BuscaResponse, ItemRecuperado, PerguntaRequest, PerguntaResponse
from .llm import ErroProvedorLLM


def obter_assistente(request: Request) -> Assistente:
    return request.app.state.assistente


# Fica no nível do módulo: com `from __future__ import annotations` o FastAPI resolve as
# anotações pelos nomes globais do módulo, e um alias local não seria encontrado.
AssistenteDep = Annotated[Assistente, Depends(obter_assistente)]


def criar_app(assistente: Assistente | None = None) -> FastAPI:
    """Fábrica da aplicação. Os testes passam um Assistente com LLM falso; em produção
    ele é montado a partir das variáveis de ambiente quando o servidor sobe."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Carregar o modelo de embeddings e indexar o cardápio uma vez só, na subida.
        app.state.assistente = assistente or criar_assistente(Config.do_ambiente())
        yield

    app = FastAPI(
        title="cardapio-rag",
        description="Perguntas sobre um cardápio fictício com RAG: busca semântica + LLM com saída em JSON.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/saude")
    def saude(assistente: AssistenteDep) -> dict:
        return {
            "status": "ok",
            "itens_indexados": len(assistente.indice.itens),
            "embeddings": assistente.indice.embedder.nome,
            "provedor": assistente.llm.nome,
            "modelo": assistente.llm.modelo,
        }

    @app.get("/buscar", response_model=BuscaResponse)
    def buscar(
        assistente: AssistenteDep,
        q: Annotated[str, Query(min_length=2, max_length=200)],
        k: Annotated[int, Query(ge=1, le=20)] = 5,
        preco_max: Annotated[float | None, Query(gt=0)] = None,
        tag: Annotated[str | None, Query(max_length=30)] = None,
    ) -> BuscaResponse:
        """Só a etapa de recuperação, sem LLM. Útil para depurar a busca."""
        resultados = assistente.indice.buscar(q, k=k, preco_max=preco_max, tag=tag)
        return BuscaResponse(
            consulta=q,
            resultados=[ItemRecuperado(item=r.item, score=r.score) for r in resultados],
        )

    @app.post("/perguntar", response_model=PerguntaResponse)
    def perguntar(corpo: PerguntaRequest, assistente: AssistenteDep) -> PerguntaResponse:
        try:
            resultado = assistente.perguntar(corpo.pergunta, k=corpo.k)
        except ErroProvedorLLM as erro:
            raise HTTPException(status_code=503, detail=f"Provedor de LLM indisponível: {erro}") from erro
        except RespostaInvalidaError as erro:
            raise HTTPException(status_code=502, detail=str(erro)) from erro

        por_id = {r.item.id: r.item for r in resultado.contexto}
        return PerguntaResponse(
            encontrado=resultado.resposta.encontrado,
            resposta=resultado.resposta.resposta,
            itens_citados=[por_id[i] for i in resultado.resposta.itens_ids],
            contexto=[ItemRecuperado(item=r.item, score=r.score) for r in resultado.contexto],
            ids_descartados=resultado.ids_descartados,
            provedor=assistente.llm.nome,
            modelo=assistente.llm.modelo,
            tentativas=resultado.tentativas,
            latencia_ms=resultado.latencia_ms,
            tempos_ms=resultado.tempos_ms,
            uso_llm=resultado.uso_llm,
        )

    return app


app = criar_app()
