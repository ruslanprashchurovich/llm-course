"""Эмбеддер: чистая математика, никакие сервисы не нужны."""

import pytest

from app.embedder import TfidfEmbedder, make_embedder

CORPUS = [
    "Каждому сотруднику положено 28 календарных дней отпуска в год.",
    "Деплой в production разрешён с понедельника по четверг.",
    "Больничный оплачивается по закону, доплата зависит от стажа.",
]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_fit_builds_vocab():
    emb = TfidfEmbedder().fit(CORPUS)
    assert emb.dim > 0
    assert len(emb.idf) == emb.dim


def test_identical_text_has_unit_similarity():
    emb = TfidfEmbedder().fit(CORPUS)
    vec = emb.embed_query(CORPUS[0])
    assert abs(dot(vec, vec) - 1.0) < 1e-9      # L2-нормировка


def test_relevant_doc_ranks_higher():
    emb = TfidfEmbedder().fit(CORPUS)
    doc_vectors = emb.embed_docs(CORPUS)
    query = emb.embed_query("сколько дней отпуска положено")
    scores = [dot(query, doc) for doc in doc_vectors]
    assert scores[0] > scores[1]                 # отпуск ближе, чем деплой
    assert scores[0] > scores[2]


def test_out_of_vocabulary_query_is_zero_vector():
    emb = TfidfEmbedder().fit(CORPUS)
    vec = emb.embed_query("жираф пингвин ксилофон")
    assert not any(vec)


def test_save_load_roundtrip(tmp_path):
    emb = TfidfEmbedder().fit(CORPUS)
    state = tmp_path / "state.json"
    emb.save(state)
    loaded = TfidfEmbedder.load(state)
    assert loaded.dim == emb.dim
    assert loaded.embed_query("отпуск") == emb.embed_query("отпуск")


# --- фабрика бэкендов (урок 3.2): e5 здесь не трогаем - он тянет модель ----------
def test_make_embedder_loads_tfidf_state(tmp_path):
    emb = TfidfEmbedder().fit(CORPUS)
    state = tmp_path / "state.json"
    emb.save(state)
    loaded = make_embedder("tfidf", state)
    assert loaded.name == "tfidf"
    assert loaded.embed_query("отпуск") == emb.embed_query("отпуск")


def test_make_embedder_rejects_unknown_backend():
    with pytest.raises(ValueError, match="tfidf"):
        make_embedder("word2vec")
