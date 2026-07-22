import pytest

from rag.ingest import Chunk
from rag.rerank import RankedChunk, aggregate_adjacent, rerank, rerank_and_aggregate


def _chunk(title="Doc", suffix="0", text="some chunk text"):
    return Chunk(chunk_id=f"{title}::{suffix}", doc_title=title, text=text)


def _ranked(title="Doc", suffix="0", text="some chunk text", cosine=0.5, rerank_score=0.5):
    return RankedChunk(chunk=_chunk(title, suffix, text), cosine_score=cosine, rerank_score=rerank_score)


def test_aggregate_adjacent_merges_consecutive_chunks_same_doc():
    a = _ranked(suffix="3", text="first half.", cosine=0.4, rerank_score=0.6)
    b = _ranked(suffix="4", text="second half.", cosine=0.5, rerank_score=0.9)
    result = aggregate_adjacent([a, b])
    assert len(result) == 1
    merged = result[0]
    assert merged.chunk.chunk_id == "Doc::3-4"
    assert merged.chunk.text == "first half. second half."
    assert merged.cosine_score == 0.5
    assert merged.rerank_score == 0.9


def test_aggregate_adjacent_does_not_merge_non_adjacent_indices():
    a = _ranked(suffix="3")
    b = _ranked(suffix="5")
    result = aggregate_adjacent([a, b])
    ids = {rc.chunk.chunk_id for rc in result}
    assert ids == {"Doc::3", "Doc::5"}


def test_aggregate_adjacent_does_not_merge_across_documents():
    a = _ranked(title="DocA", suffix="3")
    b = _ranked(title="DocB", suffix="4")
    result = aggregate_adjacent([a, b])
    ids = {rc.chunk.chunk_id for rc in result}
    assert ids == {"DocA::3", "DocB::4"}


def test_aggregate_adjacent_sorts_merged_groups_by_rerank_score_descending():
    low_group = [
        _ranked(title="Low", suffix="0", rerank_score=0.2),
        _ranked(title="Low", suffix="1", rerank_score=0.3),
    ]
    high_single = _ranked(title="High", suffix="0", rerank_score=0.9)
    result = aggregate_adjacent(low_group + [high_single])
    assert result[0].chunk.doc_title == "High"
    assert result[1].chunk.doc_title == "Low"


def test_aggregate_adjacent_empty_input_returns_empty():
    assert aggregate_adjacent([]) == []


def test_rerank_empty_candidates_returns_empty():
    assert rerank("some query", [], top_k=3) == []


def test_rerank_scores_more_relevant_chunk_higher_and_preserves_cosine_score():
    relevant = _chunk(title="PageRank", text="PageRank ranks web pages using link structure between them.")
    irrelevant = _chunk(title="Baking", suffix="1", text="Preheat the oven and mix flour with sugar.")
    # Deliberately give the irrelevant chunk the higher cosine score, so a
    # passing test proves the cross-encoder - not the input order/score -
    # is what determines the new ranking.
    candidates = [(irrelevant, 0.6), (relevant, 0.3)]

    ranked = rerank("How does PageRank rank web pages?", candidates, top_k=2)

    assert ranked[0].chunk.doc_title == "PageRank"
    assert ranked[0].cosine_score == 0.3  # original cosine score is preserved, not overwritten
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_rerank_and_aggregate_merges_reranked_adjacent_chunks():
    first_half = _chunk(title="PageRank", suffix="0", text="PageRank ranks web pages using link structure.")
    second_half = _chunk(title="PageRank", suffix="1", text="Pages linked by many important pages rank higher.")
    candidates = [(first_half, 0.5), (second_half, 0.4)]

    result = rerank_and_aggregate("How does PageRank work?", candidates, top_k=2)

    assert len(result) == 1
    assert result[0].chunk.chunk_id == "PageRank::0-1"
