import pytest

from rag.embed_store import VectorStore
from rag.ingest import Chunk


def test_query_before_build_raises_runtime_error():
    store = VectorStore()
    with pytest.raises(RuntimeError):
        store.query("anything")


def test_query_clamps_top_k_to_available_chunk_count():
    store = VectorStore()
    chunks = [
        Chunk(chunk_id="a", doc_title="A", text="cats are great pets"),
        Chunk(chunk_id="b", doc_title="B", text="dogs are loyal animals"),
    ]
    store.build(chunks)
    results = store.query("pets", top_k=10)
    assert len(results) == len(chunks)  # never returns more than exist


def test_build_and_query_ranks_most_similar_chunk_first():
    store = VectorStore()
    chunks = [
        Chunk(chunk_id="a", doc_title="PageRank", text="PageRank ranks web pages using link structure"),
        Chunk(chunk_id="b", doc_title="Baking", text="Preheat the oven and mix flour with sugar"),
    ]
    store.build(chunks)
    results = store.query("How does PageRank rank web pages?", top_k=2)
    assert results[0][0].doc_title == "PageRank"
    assert results[0][1] > results[1][1]
