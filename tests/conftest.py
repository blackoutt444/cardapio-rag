import pytest

from cardapio_rag.assistente import Assistente
from cardapio_rag.config import Config
from cardapio_rag.dados import carregar_cardapio
from cardapio_rag.embeddings import EmbedderHash
from cardapio_rag.indice import IndiceVetorial
from cardapio_rag.llm import ProvedorFake


@pytest.fixture(scope="session")
def itens():
    return carregar_cardapio(Config().cardapio_path)


@pytest.fixture(scope="session")
def indice(itens):
    # Embedder lexical: determinístico e sem download de modelo.
    return IndiceVetorial(itens, EmbedderHash())


@pytest.fixture
def fake():
    return ProvedorFake()


@pytest.fixture
def assistente(indice, fake):
    return Assistente(indice, fake, k=5)
