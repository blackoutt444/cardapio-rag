Benchmark de latência do LLM (5 perguntas por configuração)

| Configuração | Camadas na GPU | Tokens de entrada | Prompt eval | Tokens de saída | Geração | Chamada ao LLM (mediana) | JSON válido |
|---|---|---|---|---|---|---|---|
| padrão (schema, GPU automática) | 25% | 599 | 90 tok/s | 51 | 3.0 tok/s | 23.6 s | 5/5 |
| só CPU (num_gpu=0) | 0% | 599 | 52 tok/s | 48 | 6.7 tok/s | 20.6 s | 5/5 |
| sem decodificação restrita | 0% | 599 | 3587 tok/s | 48 | 7.8 tok/s | 6.4 s | 5/5 |
| contexto menor (k=3) | 0% | 460 | 492 tok/s | 47 | 7.8 tok/s | 7.4 s | 5/5 |

Ambiente da execução

- **data**: 2026-11-09T22:06:22-04:00
- **commit**: 3f64719
- **alteracoes_nao_commitadas**: False
- **python**: 3.14.5
- **sistema**: Windows-11
- **cpu**: Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
- **nucleos_logicos**: 8
- **embeddings**: intfloat/multilingual-e5-small
- **k_contexto**: 5
- **provedor**: ollama
- **modelo**: qwen2.5:3b
- **ollama_versao**: 0.40.15
- **modelo_digest**: 357c53fb659c
- **parametros**: 3.1B
- **quantizacao**: Q4_K_M
