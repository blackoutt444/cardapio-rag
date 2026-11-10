Benchmark de latência do LLM (5 perguntas por configuração)

| Configuração | Camadas na GPU | Tokens de entrada | Prompt eval | Tokens de saída | Geração | Chamada ao LLM (mediana) | JSON válido |
|---|---|---|---|---|---|---|---|
| padrão (schema, GPU automática) | 25% | 599 | 82 tok/s | 48 | 3.1 tok/s | 23.4 s | 5/5 |
| só CPU (num_gpu=0) | 0% | 599 | 50 tok/s | 48 | 8.7 tok/s | 17.7 s | 5/5 |
| sem decodificação restrita | 25% | 599 | 74 tok/s | 48 | 3.2 tok/s | 23.4 s | 5/5 |
| contexto menor (k=3) | 25% | 460 | 67 tok/s | 47 | 3.2 tok/s | 21.9 s | 5/5 |

Ambiente da execução

- **data**: 2026-11-10T00:54:07-04:00
- **commit**: b3f237d
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
