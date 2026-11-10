<h1 align="center">cardapio-rag</h1>

<p align="center">
  <b>API em Python que responde perguntas sobre um cardápio usando RAG: busca semântica + LLM com saída em JSON validado.</b><br>
  Roda 100% local e de graça (embeddings locais + Ollama), com avaliação automatizada e CI.
</p>

<p align="center">
  <a href="https://github.com/blackoutt444/cardapio-rag/actions/workflows/ci.yml"><img src="https://github.com/blackoutt444/cardapio-rag/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white" alt="Ollama">
</p>

> Projeto de estudo de engenharia de aplicações com LLM. O cardápio (4 restaurantes, 51 itens) é fictício e foi criado só para este projeto.

## O que ele faz

Você pergunta em linguagem natural ("tem hambúrguer vegano?") e a API:

1. **recupera** os itens do cardápio mais parecidos com a pergunta (embeddings + similaridade de cosseno);
2. manda esses itens como **contexto** para um LLM, com a instrução de responder só com base neles;
3. exige a resposta em um **JSON com schema definido**, valida com Pydantic e tenta de novo se vier inválido;
4. confere se o LLM só citou itens que estavam no contexto (**fidelidade**) e descarta o que não estava.

<!-- EXEMPLO -->

## Arquitetura

```mermaid
flowchart LR
    C([Cliente]) -->|POST /perguntar| API[FastAPI]
    API --> B["Recuperação<br/>embedding da pergunta<br/>+ cosseno (numpy)"]
    D[(dados/cardapio.json)] -.->|indexado na subida| B
    B -->|top-k itens| P["Prompt<br/>regras + contexto"]
    P --> L{{"ProvedorLLM<br/>Ollama · OpenAI · Anthropic · Fake"}}
    L -->|texto JSON| V["Validação Pydantic<br/>1 nova tentativa se inválido"]
    V --> F["Fidelidade<br/>ids citados ⊆ contexto"]
    F -->|resposta + itens + contexto| API
```

| Arquivo | Responsabilidade |
| --- | --- |
| [`cardapio_rag/dados.py`](cardapio_rag/dados.py) | Carrega e valida o cardápio; define o texto de cada item que vira embedding |
| [`cardapio_rag/embeddings.py`](cardapio_rag/embeddings.py) | `EmbedderE5` (semântico, sentence-transformers) e `EmbedderHash` (lexical, para testes e baseline) |
| [`cardapio_rag/indice.py`](cardapio_rag/indice.py) | Busca vetorial em memória com filtros de preço e tag |
| [`cardapio_rag/prompt.py`](cardapio_rag/prompt.py) | Prompt do sistema e formatação do contexto |
| [`cardapio_rag/llm.py`](cardapio_rag/llm.py) | Interface `ProvedorLLM` e as implementações Ollama, OpenAI, Anthropic e Fake |
| [`cardapio_rag/assistente.py`](cardapio_rag/assistente.py) | O pipeline: recuperar → gerar → validar → checar fidelidade |
| [`cardapio_rag/api.py`](cardapio_rag/api.py) | Rotas `/saude`, `/buscar` e `/perguntar` |
| [`cardapio_rag/metricas.py`](cardapio_rag/metricas.py) | hit@k e reciprocal rank |
| [`avaliacao/avaliar.py`](avaliacao/avaliar.py) | Roda as perguntas de referência e calcula as métricas |

## Como executar

Pré-requisitos: Python 3.12+ e [Ollama](https://ollama.com/download).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-embeddings.txt -r requirements-dev.txt
ollama pull qwen2.5:3b
uvicorn cardapio_rag.api:app --reload
```

Na primeira subida o modelo de embeddings (~470 MB) é baixado do Hugging Face. Depois disso, a documentação interativa fica em http://localhost:8000/docs.

```bash
curl "http://localhost:8000/buscar?q=sobremesa&preco_max=20&tag=vegano"

curl -X POST http://localhost:8000/perguntar \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Tem algum hambúrguer vegano?"}'
```

### Configuração

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `LLM_PROVEDOR` | `ollama` | `ollama`, `openai`, `anthropic` ou `fake` |
| `LLM_MODELO` | depende do provedor | `qwen2.5:3b` no Ollama |
| `OLLAMA_URL` | `http://localhost:11434` | Endereço do Ollama |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Só se usar esses provedores |
| `EMBEDDINGS` | `e5` | `e5` (semântico) ou `hash` (lexical, sem download) |
| `EMBEDDINGS_MODELO` | `intfloat/multilingual-e5-small` | Qualquer modelo E5 do Hugging Face |
| `TOP_K` | `5` | Itens recuperados como contexto |

### Docker

```bash
docker build -t cardapio-rag .
docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway cardapio-rag
```

O container fala com o Ollama da máquina host. Para testar sem LLM nem download de modelo: `-e LLM_PROVEDOR=fake -e EMBEDDINGS=hash`.

### Testes

```bash
pytest -v
ruff check .
```

Os testes não usam rede nem modelo: o LLM é o `ProvedorFake`, os embeddings são os lexicais (`EmbedderHash`) e as APIs da OpenAI, da Anthropic e do Ollama são simuladas com `httpx.MockTransport`. O CI roda lint, testes (Python 3.12 e 3.13), a avaliação da busca lexical e o build da imagem Docker com um teste de fumaça em `/saude`.

## Avaliação

<!-- RESULTADOS -->

## Decisões técnicas

- **RAG em vez de só o LLM.** O modelo não conhece este cardápio, e colocar o cardápio inteiro no prompt não escala. Recuperar só os itens relevantes deixa o prompt curto (importante num modelo de 3B rodando em CPU) e torna a resposta rastreável até o item de origem.
- **Um item do cardápio = um chunk.** O dado já é estruturado e curto; quebrar em pedaços menores só separaria o nome do prato da descrição. O preço fica fora do texto do embedding e entra como filtro (`preco_max`), porque número não ajuda na similaridade semântica.
- **`multilingual-e5-small`.** É pequeno (~118M parâmetros, roda em CPU), entende português e a família E5 tem bons resultados em recuperação. Ela exige os prefixos `query:` e `passage:`; sem eles a qualidade cai.
- **numpy em vez de banco vetorial.** Com 51 itens, comparar a pergunta com todos (busca exata) leva microssegundos. Vetores normalizados fazem o produto escalar ser a similaridade de cosseno. Um banco vetorial (pgvector, Qdrant, FAISS) entraria com dezenas de milhares de itens ou com necessidade de persistência.
- **Saída estruturada em duas camadas.** O JSON Schema do modelo Pydantic `RespostaLLM` vai para o provedor (campo `format` no Ollama, `json_schema` estrito na OpenAI, ferramenta obrigatória na Anthropic), o que restringe a geração. Mesmo assim a resposta é validada com Pydantic, e um JSON inválido gera uma nova tentativa com a mensagem de erro. Se falhar de novo, a API responde 502 em vez de devolver lixo.
- **Fidelidade checada no código.** O LLM precisa citar os ids dos itens que usou. Ids que não estavam no contexto são descartados e devolvidos em `ids_descartados`, para não mostrar ao cliente um prato que não existe.
- **Abstenção explícita.** O campo `encontrado` obriga o modelo a dizer quando o cardápio não tem a resposta, e a avaliação mede isso com perguntas sem resposta de propósito (sushi, feijoada, horário de funcionamento).
- **Interface de provedor.** O pipeline só conhece `ProvedorLLM.gerar_json()`. Trocar de LLM é mudar `LLM_PROVEDOR`. As chamadas usam `httpx` direto em vez dos SDKs: são poucas linhas, não há dependência extra e dá para testar o formato das requisições sem chave.
- **`temperature=0` no Ollama**, para a avaliação ser reproduzível.
- **CI sem PyTorch.** `sentence-transformers` fica em `requirements-embeddings.txt`, importado só quando o `EmbedderE5` é usado. O CI instala só o necessário e roda em segundos.

## Limitações

- **Dados e perguntas pequenos e escritos por mim.** São 51 itens e 31 perguntas. Os números mostram tendência, não têm significância estatística, e as perguntas podem ter o viés de quem conhece o cardápio.
- **A métrica de fidelidade só olha os ids.** Ela não pega um preço ou ingrediente inventado dentro do texto da resposta. Um próximo passo seria checar os preços citados contra o item ou usar um LLM como juiz.
- **Perguntas de agregação** ("qual o prato vegano mais barato?") dependem de o item certo estar entre os k recuperados. Para isso, um filtro estruturado ou uma ferramenta de consulta seriam melhores que busca semântica.
- **Latência.** Um modelo de 3B em CPU de notebook leva segundos por resposta. Com GPU ou com um provedor em nuvem cai bastante.
- **OpenAI e Anthropic** foram testados só com respostas simuladas, porque não usei chave paga.
- Sem histórico de conversa, autenticação ou limite de requisições.

## Como foi desenvolvido

Esta seção trata do processo de desenvolvimento, não do que a aplicação faz.

O código deste repositório, inclusive as métricas de avaliação e os testes, foi escrito por um agente de programação (Claude Code) a partir de uma especificação minha. Eu defini o escopo, as restrições (rodar sem chave de API paga, LLM local, resultados reproduzíveis), o framework e os critérios das métricas, e faço a revisão técnica de cada mudança nos pull requests. Os commits escritos pelo agente têm a linha `Co-Authored-By` no histórico.

Usar um agente de programação foi uma forma de desenvolver o projeto. O que o projeto demonstra como aplicação com LLM está descrito nas seções acima: recuperação com embeddings, geração com saída estruturada e validada, checagem de fidelidade e avaliação.
