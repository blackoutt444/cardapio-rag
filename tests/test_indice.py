def test_busca_encontra_item_pelo_ingrediente(indice):
    resultados = indice.buscar("Tem algum prato com pirarucu?", k=3)
    assert resultados[0].item.id == "SR04"


def test_busca_respeita_k_e_ordena_por_score(indice):
    resultados = indice.buscar("hambúrguer", k=4)
    assert len(resultados) == 4
    scores = [r.score for r in resultados]
    assert scores == sorted(scores, reverse=True)


def test_filtro_preco_max(indice):
    resultados = indice.buscar("sobremesa", k=10, preco_max=20)
    assert resultados
    assert all(r.item.preco <= 20 for r in resultados)


def test_filtro_tag_ignora_acento_e_maiusculas(indice):
    resultados = indice.buscar("massa", k=10, tag="Sem Gluten")
    assert resultados
    assert all("sem glúten" in r.item.tags for r in resultados)


def test_filtro_sem_resultados(indice):
    assert indice.buscar("pizza", preco_max=1) == []
