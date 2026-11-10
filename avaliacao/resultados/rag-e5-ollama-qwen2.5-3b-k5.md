Recuperação (25 perguntas com resposta no cardápio)

| Embeddings | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | 92% | 96% | 100% | 0.95 |

Geração com `ollama/qwen2.5:3b` (31 perguntas)

| Métrica | Resultado |
|---|---|
| JSON válido na 1ª tentativa | 100% |
| JSON válido (com até 1 nova tentativa) | 100% |
| Fidelidade (nenhum id fora do contexto) | 100% |
| Acerto nas perguntas com resposta (25) | 96% |
| Abstenção correta nas sem resposta (6) | 100% |
| Precisão das citações (24 respostas com citação) | 96% |
| Preços no texto conferem com o cardápio (7 respostas com preço) | 100% |

Latência com o modelo já carregado (30 perguntas)

| Etapa | Mediana | p90 |
|---|---|---|
| Embedding da pergunta | 31 ms | 50 ms |
| Busca (cosseno + filtros + ordenação) | 0.12 ms | 0.17 ms |
| Montagem do prompt | 0.05 ms | 0.07 ms |
| Chamada ao LLM (inclui validação) | 26.5 s | 32.2 s |
| &nbsp;&nbsp;↳ carga do modelo | 13 ms | 16 ms |
| &nbsp;&nbsp;↳ processamento do prompt (prompt eval) | 10.8 s | 13.1 s |
| &nbsp;&nbsp;↳ geração da resposta | 15.2 s | 21.6 s |
| **Total** | 26.5 s | 32.3 s |

Tokens por pergunta (mediana): 582 de entrada, 46 de saída. Vazão: 53 tokens/s no processamento do prompt e 3.0 tokens/s na geração.

Primeira pergunta (modelo frio): 36.0 s no total, dos quais 7.3 s para carregar o modelo.

Ambiente da execução

- **data**: 2026-11-09T22:06:57-04:00
- **commit**: 3f64719
- **alteracoes_nao_commitadas**: False
- **python**: 3.14.5
- **sistema**: Windows-11
- **cpu**: Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
- **nucleos_logicos**: 8
- **embeddings**: intfloat/multilingual-e5-small
- **k_contexto**: 5
- **carga_embedder_s**: 21.7
- **indexacao_s**: 1.59
- **provedor**: ollama
- **modelo**: qwen2.5:3b
- **ollama_versao**: 0.40.15
- **modelo_digest**: 357c53fb659c
- **parametros**: 3.1B
- **quantizacao**: Q4_K_M
- **fracao_em_gpu**: 0.25
