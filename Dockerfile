FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# PyTorch só de CPU: a versão padrão do PyPI no Linux traz CUDA e deixa a imagem
# vários GB maior. Fica numa camada própria para ser reaproveitada no cache.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-embeddings.txt ./
RUN pip install -r requirements-embeddings.txt

COPY cardapio_rag ./cardapio_rag
COPY dados ./dados

# Usuário sem privilégios; o cache do Hugging Face (modelo de embeddings) fica no home dele.
RUN useradd --create-home app
USER app

# O Ollama roda fora do container (na máquina host).
ENV OLLAMA_URL=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "cardapio_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
