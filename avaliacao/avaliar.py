"""Avaliação do RAG com as perguntas de referência (avaliacao/perguntas.json).

Recuperação (só perguntas que têm resposta no cardápio):
  hit@k  fração de perguntas com pelo menos um item esperado entre os k primeiros
  MRR    média de 1/posição do primeiro item esperado

Geração (todas as perguntas, uma chamada ao LLM por pergunta):
  JSON válido        resposta passou na validação do Pydantic (na 1ª tentativa / com retry)
  Fidelidade         resposta não citou nenhum id fora do contexto recuperado
  Acerto             pergunta com resposta: encontrado=true e citou um item esperado
  Abstenção correta  pergunta sem resposta: encontrado=false

Cada execução salva em avaliacao/resultados/ um .json (métricas, metadados e o
resultado de cada pergunta) e um .md (as tabelas). Os metadados registram data,
commit, hardware e versão exata do modelo, para o resultado ser reproduzível.

Uso:
  python -m avaliacao.avaliar --somente-busca --embeddings hash
  python -m avaliacao.avaliar --embeddings e5 --provedor ollama
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime

import httpx

from cardapio_rag.assistente import Assistente, RespostaInvalidaError
from cardapio_rag.config import RAIZ, Config
from cardapio_rag.dados import carregar_cardapio
from cardapio_rag.embeddings import criar_embedder
from cardapio_rag.indice import IndiceVetorial
from cardapio_rag.llm import criar_provedor
from cardapio_rag.metricas import hit_at_k, media, reciprocal_rank

KS = (1, 3, 5)
PASTA_RESULTADOS = RAIZ / "avaliacao" / "resultados"


def pct(valor: float) -> str:
    return f"{valor * 100:.0f}%"


# ---------- Avaliação ----------


def avaliar_recuperacao(indice: IndiceVetorial, perguntas: list[dict]) -> tuple[dict, list[dict]]:
    respondiveis = [p for p in perguntas if p["esperados"]]
    detalhes = []
    for p in respondiveis:
        ids = [r.item.id for r in indice.buscar(p["pergunta"], k=max(KS))]
        esperados = set(p["esperados"])
        detalhes.append(
            {
                "id": p["id"],
                "pergunta": p["pergunta"],
                "esperados": p["esperados"],
                "recuperados": ids,
                **{f"hit@{k}": hit_at_k(ids, esperados, k) for k in KS},
                "rr": reciprocal_rank(ids, esperados),
            }
        )
    resumo = {f"hit@{k}": media([d[f"hit@{k}"] for d in detalhes]) for k in KS}
    resumo["mrr"] = media([d["rr"] for d in detalhes])
    resumo["n"] = len(detalhes)
    return resumo, detalhes


def avaliar_geracao(assistente: Assistente, perguntas: list[dict]) -> tuple[dict, list[dict]]:
    detalhes = []
    for n, p in enumerate(perguntas, 1):
        print(f"  [{n}/{len(perguntas)}] {p['pergunta']}", file=sys.stderr, flush=True)
        esperados = set(p["esperados"])
        registro = {"id": p["id"], "pergunta": p["pergunta"], "esperados": p["esperados"]}
        try:
            r = assistente.perguntar(p["pergunta"])
        except RespostaInvalidaError as erro:
            registro.update(valido=False, erro=str(erro))
            detalhes.append(registro)
            continue
        citados = set(r.resposta.itens_ids)
        registro.update(
            valido=True,
            tentativas=r.tentativas,
            encontrado=r.resposta.encontrado,
            resposta=r.resposta.resposta,
            itens_ids=r.resposta.itens_ids,
            ids_descartados=r.ids_descartados,
            contexto=[c.item.id for c in r.contexto],
            fiel=not r.ids_descartados,
            correto=(
                r.resposta.encontrado and bool(citados & esperados)
                if esperados
                else not r.resposta.encontrado
            ),
            tempos_ms=r.tempos_ms,
            uso_llm=r.uso_llm,
        )
        detalhes.append(registro)

    validos = [d for d in detalhes if d["valido"]]
    respondiveis = [d for d in detalhes if d["esperados"]]
    sem_resposta = [d for d in detalhes if not d["esperados"]]
    resumo = {
        "n": len(detalhes),
        "json_valido_1a": media([float(d.get("tentativas") == 1) for d in detalhes]),
        "json_valido": media([float(d["valido"]) for d in detalhes]),
        "fidelidade": media([float(d["fiel"]) for d in validos]),
        "acerto": media([float(d.get("correto", False)) for d in respondiveis]),
        "abstencao": media([float(d.get("correto", False)) for d in sem_resposta]),
        "n_respondiveis": len(respondiveis),
        "n_sem_resposta": len(sem_resposta),
    }
    return resumo, detalhes


# ---------- Latência ----------

ETAPAS = ("embedding", "busca", "prompt", "llm", "total")
ETAPAS_LLM = ("carga_s", "prompt_eval_s", "geracao_s")


def p90(valores: list[float]) -> float:
    if len(valores) < 2:
        return valores[0]
    return statistics.quantiles(valores, n=10, method="inclusive")[-1]


def resumir_latencia(detalhes: list[dict]) -> dict:
    """A 1ª pergunta paga a carga do modelo na memória (início a frio); as demais medem
    o regime normal (modelo já carregado). Por isso são resumidas separadamente."""
    validos = [d for d in detalhes if d["valido"]]
    if not validos:
        return {}
    frio, quentes = validos[0], (validos[1:] or validos)

    etapas = {}
    for etapa in ETAPAS:
        valores = [d["tempos_ms"][etapa] for d in quentes]
        etapas[etapa] = {"mediana_ms": statistics.median(valores), "p90_ms": p90(valores)}
    for campo in ETAPAS_LLM:
        valores = [d["uso_llm"][campo] * 1000 for d in quentes if d["uso_llm"][campo] is not None]
        if valores:
            etapas[campo.removesuffix("_s")] = {
                "mediana_ms": statistics.median(valores),
                "p90_ms": p90(valores),
            }

    resumo: dict = {
        "n_quentes": len(quentes),
        "quente": etapas,
        "frio": {"total_ms": frio["tempos_ms"]["total"], **frio["uso_llm"]},
    }

    def soma(campo: str) -> float:
        return sum(d["uso_llm"][campo] or 0 for d in quentes)

    if soma("tokens_entrada"):
        resumo["tokens"] = {
            "entrada_mediana": statistics.median(d["uso_llm"]["tokens_entrada"] or 0 for d in quentes),
            "saida_mediana": statistics.median(d["uso_llm"]["tokens_saida"] or 0 for d in quentes),
        }
        if soma("prompt_eval_s") and soma("geracao_s"):
            resumo["tokens"]["prompt_eval_tok_s"] = soma("tokens_entrada") / soma("prompt_eval_s")
            resumo["tokens"]["geracao_tok_s"] = soma("tokens_saida") / soma("geracao_s")
    return resumo


def descarregar_modelo_ollama(url: str, modelo: str) -> None:
    """keep_alive=0 tira o modelo da memória, para a 1ª pergunta medir o início a frio."""
    try:
        httpx.post(f"{url}/api/generate", json={"model": modelo, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError as erro:
        print(f"Aviso: não consegui descarregar o modelo ({erro})", file=sys.stderr)


# ---------- Metadados (evidência) ----------


def _git(*args: str) -> str | None:
    try:
        saida = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return saida.stdout.strip()


def nome_cpu() -> str:
    if sys.platform == "win32":
        import winreg

        chave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(chave, "ProcessorNameString")[0].strip()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linha in f:
                if linha.startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconhecida"


def metadados_execucao(config: Config, nome_embedder: str) -> dict:
    meta = {
        "data": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "alteracoes_nao_commitadas": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "sistema": platform.platform(terse=True),
        "cpu": nome_cpu(),
        "nucleos_logicos": os.cpu_count(),
        "embeddings": nome_embedder,
        "k_contexto": config.top_k,
    }
    if os.getenv("GITHUB_ACTIONS") == "true":
        servidor, repo, run = (
            os.getenv(v) for v in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
        )
        meta["ci_run"] = f"{servidor}/{repo}/actions/runs/{run}"
    return meta


def metadados_ollama(url: str, modelo: str) -> dict:
    """Versão do Ollama, digest e quantização do modelo, e se ele rodou em CPU ou GPU."""
    try:
        versao = httpx.get(f"{url}/api/version", timeout=10).json()["version"]
        tags = httpx.get(f"{url}/api/tags", timeout=10).json()["models"]
        carregados = httpx.get(f"{url}/api/ps", timeout=10).json()["models"]
    except (httpx.HTTPError, KeyError, ValueError) as erro:
        return {"erro": str(erro)}
    info = next((m for m in tags if m["name"] == modelo), {})
    detalhes = info.get("details", {})
    meta = {
        "ollama_versao": versao,
        "modelo_digest": info.get("digest", "")[:12],
        "parametros": detalhes.get("parameter_size"),
        "quantizacao": detalhes.get("quantization_level"),
    }
    em_uso = next((m for m in carregados if m["name"] == modelo), None)
    if em_uso and em_uso.get("size"):
        meta["fracao_em_gpu"] = round(em_uso.get("size_vram", 0) / em_uso["size"], 2)
    return meta


# ---------- Saída ----------


def markdown_recuperacao(rec: dict, nome_embedder: str) -> str:
    linhas = [
        f"| Embeddings | {' | '.join(f'hit@{k}' for k in KS)} | MRR |",
        "|---|" + "---|" * (len(KS) + 1),
        f"| `{nome_embedder}` | {' | '.join(pct(rec[f'hit@{k}']) for k in KS)} | {rec['mrr']:.2f} |",
    ]
    return f"Recuperação ({rec['n']} perguntas com resposta no cardápio)\n\n" + "\n".join(linhas)


def markdown_geracao(ger: dict, rotulo_modelo: str) -> str:
    linhas = [
        "| Métrica | Resultado |",
        "|---|---|",
        f"| JSON válido na 1ª tentativa | {pct(ger['json_valido_1a'])} |",
        f"| JSON válido (com até 1 nova tentativa) | {pct(ger['json_valido'])} |",
        f"| Fidelidade (nenhum id fora do contexto) | {pct(ger['fidelidade'])} |",
        f"| Acerto nas perguntas com resposta ({ger['n_respondiveis']}) | {pct(ger['acerto'])} |",
        f"| Abstenção correta nas sem resposta ({ger['n_sem_resposta']}) | {pct(ger['abstencao'])} |",
    ]
    return f"Geração com `{rotulo_modelo}` ({ger['n']} perguntas)\n\n" + "\n".join(linhas)


def fmt_ms(valor: float | None) -> str:
    if valor is None:
        return "—"
    if valor >= 1000:
        return f"{valor / 1000:.1f} s"
    if valor >= 1:
        return f"{valor:.0f} ms"
    return f"{valor:.2f} ms"


def markdown_latencia(lat: dict) -> str:
    rotulos = {
        "embedding": "Embedding da pergunta",
        "busca": "Busca (cosseno + filtros + ordenação)",
        "prompt": "Montagem do prompt",
        "llm": "Chamada ao LLM (inclui validação)",
        "carga": "&nbsp;&nbsp;↳ carga do modelo",
        "prompt_eval": "&nbsp;&nbsp;↳ processamento do prompt (prompt eval)",
        "geracao": "&nbsp;&nbsp;↳ geração da resposta",
        "total": "**Total**",
    }
    q = lat["quente"]
    linhas = ["| Etapa | Mediana | p90 |", "|---|---|---|"]
    for chave in ("embedding", "busca", "prompt", "llm", "carga", "prompt_eval", "geracao", "total"):
        if chave in q:
            linhas.append(
                f"| {rotulos[chave]} | {fmt_ms(q[chave]['mediana_ms'])} | {fmt_ms(q[chave]['p90_ms'])} |"
            )
    texto = f"Latência com o modelo já carregado ({lat['n_quentes']} perguntas)\n\n" + "\n".join(linhas)

    if "tokens" in lat:
        t = lat["tokens"]
        texto += (
            f"\n\nTokens por pergunta (mediana): {t['entrada_mediana']:.0f} de entrada,"
            f" {t['saida_mediana']:.0f} de saída."
        )
        if "geracao_tok_s" in t:
            texto += (
                f" Vazão: {t['prompt_eval_tok_s']:.0f} tokens/s no processamento do prompt"
                f" e {t['geracao_tok_s']:.1f} tokens/s na geração."
            )
    frio = lat["frio"]
    texto += f"\n\nPrimeira pergunta (modelo frio): {fmt_ms(frio['total_ms'])} no total"
    if frio.get("carga_s") is not None:
        texto += f", dos quais {fmt_ms(frio['carga_s'] * 1000)} para carregar o modelo"
    return texto + "."


def markdown_metadados(meta: dict) -> str:
    itens = [f"- **{chave}**: {valor}" for chave, valor in meta.items() if valor not in (None, "")]
    return "Ambiente da execução\n\n" + "\n".join(itens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--embeddings", choices=["e5", "hash"], help="padrão: variável EMBEDDINGS ou e5")
    parser.add_argument("--provedor", help="padrão: variável LLM_PROVEDOR ou ollama")
    parser.add_argument("--modelo", help="padrão: modelo padrão do provedor")
    parser.add_argument("--k", type=int, default=5, help="itens no contexto do LLM")
    parser.add_argument("--somente-busca", action="store_true", help="avalia só a recuperação, sem LLM")
    args = parser.parse_args()

    trocas: dict = {"top_k": args.k}
    if args.embeddings:
        trocas["embeddings"] = args.embeddings
    if args.provedor:
        trocas["llm_provedor"] = args.provedor
    if args.modelo:
        trocas["llm_modelo"] = args.modelo
    config = dataclasses.replace(Config.do_ambiente(), **trocas)

    perguntas = json.loads((RAIZ / "avaliacao" / "perguntas.json").read_text(encoding="utf-8"))["perguntas"]
    t0 = time.perf_counter()
    embedder = criar_embedder(config)
    t1 = time.perf_counter()
    indice = IndiceVetorial(carregar_cardapio(config.cardapio_path), embedder)
    t2 = time.perf_counter()
    meta = metadados_execucao(config, embedder.nome)
    # Custo de subida da API: carregar o modelo de embeddings e indexar o cardápio.
    meta["carga_embedder_s"] = round(t1 - t0, 2)
    meta["indexacao_s"] = round(t2 - t1, 2)

    rec, rec_detalhes = avaliar_recuperacao(indice, perguntas)
    secoes = [markdown_recuperacao(rec, embedder.nome)]
    saida: dict = {"metadados": meta, "recuperacao": rec, "recuperacao_detalhes": rec_detalhes}
    nome_arquivo = f"busca-{config.embeddings}"

    if not args.somente_busca:
        llm = criar_provedor(config)
        if llm.nome == "ollama":
            descarregar_modelo_ollama(config.ollama_url, llm.modelo)
        print(f"Gerando respostas com {llm.nome}/{llm.modelo}...", file=sys.stderr)
        ger, ger_detalhes = avaliar_geracao(Assistente(indice, llm, k=config.top_k), perguntas)
        lat = resumir_latencia(ger_detalhes)
        meta.update(provedor=llm.nome, modelo=llm.modelo)
        if llm.nome == "ollama":
            meta.update(metadados_ollama(config.ollama_url, llm.modelo))
        secoes.append(markdown_geracao(ger, f"{llm.nome}/{llm.modelo}"))
        if lat:
            secoes.append(markdown_latencia(lat))
        saida.update(geracao=ger, latencia=lat, geracao_detalhes=ger_detalhes)
        nome = f"rag-{config.embeddings}-{llm.nome}-{llm.modelo}-k{config.top_k}"
        nome_arquivo = nome.replace(":", "-").replace("/", "-")

    secoes.append(markdown_metadados(meta))
    markdown = "\n\n".join(secoes) + "\n"
    print("\n" + markdown)

    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as resumo:
            resumo.write(f"## Avaliação: {nome_arquivo}\n\n{markdown}\n")

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    (PASTA_RESULTADOS / f"{nome_arquivo}.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (PASTA_RESULTADOS / f"{nome_arquivo}.md").write_text(markdown, encoding="utf-8")
    print(f"Resultados salvos em avaliacao/resultados/{nome_arquivo}.json e .md", file=sys.stderr)


if __name__ == "__main__":
    main()
