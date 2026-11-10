"""Benchmark de latência do LLM no Ollama.

A avaliação mostra que mais de 99% do tempo de uma resposta está na chamada ao LLM.
Este script isola o que pesa nessa chamada: roda as mesmas perguntas em
configurações diferentes e compara a vazão (tokens/s) do processamento do prompt
e da geração.

  padrão          JSON Schema no "format" e o Ollama decide quantas camadas vão para a GPU
  só CPU          num_gpu=0: nenhuma camada na GPU
  sem schema      sem decodificação restrita (o prompt continua pedindo JSON)
  k=3             contexto com 3 itens em vez de 5 (menos tokens de entrada)

Antes de cada configuração o modelo é descarregado (keep_alive=0). Isso garante
que cada uma começa com a divisão CPU/GPU padrão do Ollama (só a linha "só CPU" a
muda) e que o cache de prompt da configuração anterior não é reaproveitado: sem
isso, repetir as mesmas perguntas faria o processamento do prompt parecer
instantâneo. Depois vem uma chamada de aquecimento, descartada, que absorve a carga
do modelo e deixa em cache só o prompt de sistema, igual para todas as configurações.

Uso:
  python -m avaliacao.latencia [--repeticoes 5]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time

import httpx
from pydantic import ValidationError

from avaliacao.avaliar import (
    PASTA_RESULTADOS,
    descarregar_modelo_ollama,
    metadados_execucao,
    metadados_ollama,
)
from cardapio_rag.assistente import SCHEMA_RESPOSTA
from cardapio_rag.config import RAIZ, Config
from cardapio_rag.dados import carregar_cardapio
from cardapio_rag.embeddings import criar_embedder
from cardapio_rag.esquemas import RespostaLLM
from cardapio_rag.indice import IndiceVetorial
from cardapio_rag.llm import ProvedorOllama
from cardapio_rag.prompt import SISTEMA, montar_mensagem_usuario

CONFIGURACOES = [
    ("padrão (schema, GPU automática)", {"schema": True, "opcoes": {}, "k": 5}),
    ("só CPU (num_gpu=0)", {"schema": True, "opcoes": {"num_gpu": 0}, "k": 5}),
    ("sem decodificação restrita", {"schema": False, "opcoes": {}, "k": 5}),
    ("contexto menor (k=3)", {"schema": True, "opcoes": {}, "k": 3}),
]
# Fora do conjunto medido: se fosse uma das perguntas, ela inteira ficaria em cache.
AQUECIMENTO = "Tem algum prato com camarão?"


def fracao_em_gpu(url: str, modelo: str) -> float | None:
    try:
        carregados = httpx.get(f"{url}/api/ps", timeout=10).json()["models"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    m = next((m for m in carregados if m["name"] == modelo), None)
    return round(m.get("size_vram", 0) / m["size"], 2) if m and m.get("size") else None


def medir(provedor: ProvedorOllama, mensagem: str, schema: bool) -> dict:
    inicio = time.perf_counter()
    g = provedor.gerar_json(SISTEMA, mensagem, SCHEMA_RESPOSTA if schema else {})
    chamada_s = time.perf_counter() - inicio
    try:
        RespostaLLM.model_validate_json(g.texto)
        valido = True
    except ValidationError:
        valido = False
    return {
        "chamada_s": chamada_s,
        "tokens_entrada": g.tokens_entrada,
        "prompt_eval_s": g.prompt_eval_s,
        "tokens_saida": g.tokens_saida,
        "geracao_s": g.geracao_s,
        "carga_s": g.carga_s,
        "json_valido": valido,
    }


def resumir(medidas: list[dict]) -> dict:
    def soma(campo: str) -> float:
        return sum(m[campo] or 0 for m in medidas)

    return {
        "n": len(medidas),
        "tokens_entrada_mediana": statistics.median(m["tokens_entrada"] for m in medidas),
        "tokens_saida_mediana": statistics.median(m["tokens_saida"] for m in medidas),
        "prompt_eval_tok_s": soma("tokens_entrada") / soma("prompt_eval_s"),
        "geracao_tok_s": soma("tokens_saida") / soma("geracao_s"),
        "chamada_mediana_s": statistics.median(m["chamada_s"] for m in medidas),
        "json_valido": sum(m["json_valido"] for m in medidas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repeticoes", type=int, default=5, help="perguntas medidas por configuração")
    args = parser.parse_args()

    config = dataclasses.replace(Config.do_ambiente(), llm_provedor="ollama", embeddings="e5")
    modelo = config.llm_modelo or "qwen2.5:3b"
    indice = IndiceVetorial(carregar_cardapio(config.cardapio_path), criar_embedder(config))
    perguntas = json.loads((RAIZ / "avaliacao" / "perguntas.json").read_text(encoding="utf-8"))["perguntas"]
    perguntas = [p["pergunta"] for p in perguntas[: args.repeticoes]]
    assert AQUECIMENTO not in perguntas, "a pergunta de aquecimento não pode estar entre as medidas"
    # Lido antes das medições para registrar o commit que foi de fato medido.
    meta = metadados_execucao(config, indice.embedder.nome)

    resultados = []
    for nome, cfg in CONFIGURACOES:
        provedor = ProvedorOllama(modelo, config.ollama_url, config.llm_timeout_s, opcoes=cfg["opcoes"])

        def mensagem(pergunta: str, k: int = cfg["k"]) -> str:
            return montar_mensagem_usuario(pergunta, indice.buscar(pergunta, k=k))

        print(f"{nome}: descarregando o modelo e aquecendo...", file=sys.stderr, flush=True)
        descarregar_modelo_ollama(config.ollama_url, modelo)
        medir(provedor, mensagem(AQUECIMENTO), cfg["schema"])
        medidas = []
        for i, pergunta in enumerate(perguntas, 1):
            print(f"  [{i}/{len(perguntas)}] {pergunta}", file=sys.stderr, flush=True)
            medidas.append(medir(provedor, mensagem(pergunta), cfg["schema"]))
        resultados.append(
            {
                "configuracao": nome,
                **cfg,
                "fracao_em_gpu": fracao_em_gpu(config.ollama_url, modelo),
                "resumo": resumir(medidas),
                "medidas": medidas,
            }
        )

    meta.update(provedor="ollama", modelo=modelo, **metadados_ollama(config.ollama_url, modelo))
    meta.pop("fracao_em_gpu", None)  # varia por configuração; está em cada linha da tabela

    linhas = [
        "| Configuração | Camadas na GPU | Tokens de entrada | Prompt eval | Tokens de saída | Geração "
        "| Chamada ao LLM (mediana) | JSON válido |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in resultados:
        s = r["resumo"]
        gpu = "—" if r["fracao_em_gpu"] is None else f"{r['fracao_em_gpu'] * 100:.0f}%"
        celulas = [
            r["configuracao"],
            gpu,
            f"{s['tokens_entrada_mediana']:.0f}",
            f"{s['prompt_eval_tok_s']:.0f} tok/s",
            f"{s['tokens_saida_mediana']:.0f}",
            f"{s['geracao_tok_s']:.1f} tok/s",
            f"{s['chamada_mediana_s']:.1f} s",
            f"{s['json_valido']}/{s['n']}",
        ]
        linhas.append("| " + " | ".join(celulas) + " |")
    ambiente = "\n".join(f"- **{c}**: {v}" for c, v in meta.items() if v not in (None, ""))
    markdown = (
        f"Benchmark de latência do LLM ({args.repeticoes} perguntas por configuração)\n\n"
        + "\n".join(linhas)
        + f"\n\nAmbiente da execução\n\n{ambiente}\n"
    )
    print("\n" + markdown)

    nome_arquivo = f"latencia-ollama-{modelo}".replace(":", "-").replace("/", "-")
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    (PASTA_RESULTADOS / f"{nome_arquivo}.json").write_text(
        json.dumps({"metadados": meta, "configuracoes": resultados}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (PASTA_RESULTADOS / f"{nome_arquivo}.md").write_text(markdown, encoding="utf-8")
    print(f"Resultados salvos em avaliacao/resultados/{nome_arquivo}.json e .md", file=sys.stderr)


if __name__ == "__main__":
    main()
