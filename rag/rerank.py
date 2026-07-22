"""
Reranking + aggregation: narrow a wide candidate pool down to the best,
non-fragmented passages before generation.

Cross-encoder reranking scores each (query, chunk) pair jointly instead of
comparing independently-embedded vectors, which is more accurate than cosine
similarity for judging *specific* relevance (see EVALUATION.md's rerank
comparison — it fixes a documented miss where a generic placeholder-doc
chunk outranked the actually-correct, more specific lecture chunk).
Aggregation then stitches together any adjacent chunks from the same
document among the reranked top-k into one contiguous passage, since
consecutive fragments read better merged than as separate cards.

The relevance floor and `generate.py` both keep operating on cosine
similarity, unchanged — the cross-encoder only decides ranking order and
which chunks survive the narrowing to top_k, not what counts as "relevant."
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from sentence_transformers import CrossEncoder

from .ingest import Chunk

_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def _get_reranker() -> CrossEncoder:
    """Lazily load the cross-encoder once and share it across calls."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANK_MODEL_NAME, default_activation_function=torch.nn.Sigmoid())
    return _reranker


@dataclass
class RankedChunk:
    chunk: Chunk
    cosine_score: float
    rerank_score: float


def rerank(query: str, candidates: List[Tuple[Chunk, float]], top_k: int) -> List[RankedChunk]:
    """Re-score candidates with a cross-encoder and keep the best top_k.

    Each result keeps its original cosine score alongside the new rerank
    score, so the relevance floor and `generate.py` can keep operating on
    the cosine score unchanged - the cross-encoder only affects order and
    which chunks survive the cut to top_k.
    """
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, chunk.text) for chunk, _ in candidates]
    rerank_scores = model.predict(pairs)
    ranked = [
        RankedChunk(chunk=chunk, cosine_score=cosine_score, rerank_score=float(score))
        for (chunk, cosine_score), score in zip(candidates, rerank_scores)
    ]
    ranked.sort(key=lambda rc: rc.rerank_score, reverse=True)
    return ranked[:top_k]


def _chunk_index(chunk_id: str) -> int:
    return int(chunk_id.rsplit("::", 1)[-1])


def _merge_run(doc_title: str, run: List[Tuple[int, RankedChunk]]) -> RankedChunk:
    start_idx, end_idx = run[0][0], run[-1][0]
    chunk_id = f"{doc_title}::{start_idx}" if start_idx == end_idx else f"{doc_title}::{start_idx}-{end_idx}"
    text = " ".join(rc.chunk.text for _, rc in run)
    merged_chunk = Chunk(chunk_id=chunk_id, doc_title=doc_title, text=text)
    return RankedChunk(
        chunk=merged_chunk,
        cosine_score=max(rc.cosine_score for _, rc in run),
        rerank_score=max(rc.rerank_score for _, rc in run),
    )


def aggregate_adjacent(ranked: List[RankedChunk]) -> List[RankedChunk]:
    """Merge strictly-consecutive same-document chunks among the reranked
    results into single passages, then re-sort by rerank score.

    Chunks are only merged when both are already present in `ranked` and
    their indices are consecutive (e.g. 3 and 4) - a gap (e.g. 3 and 5,
    without 4) is left as two separate entries rather than pulling in a
    chunk that wasn't actually retrieved.
    """
    by_doc = {}
    for rc in ranked:
        idx = _chunk_index(rc.chunk.chunk_id)
        by_doc.setdefault(rc.chunk.doc_title, []).append((idx, rc))

    merged: List[RankedChunk] = []
    for doc_title, entries in by_doc.items():
        entries.sort(key=lambda e: e[0])
        run = [entries[0]]
        for idx, rc in entries[1:]:
            if idx == run[-1][0] + 1:
                run.append((idx, rc))
            else:
                merged.append(_merge_run(doc_title, run))
                run = [(idx, rc)]
        merged.append(_merge_run(doc_title, run))

    merged.sort(key=lambda rc: rc.rerank_score, reverse=True)
    return merged


def rerank_and_aggregate(query: str, candidates: List[Tuple[Chunk, float]], top_k: int) -> List[RankedChunk]:
    """Rerank candidates with the cross-encoder, then merge adjacent
    same-document chunks among the results into contiguous passages."""
    return aggregate_adjacent(rerank(query, candidates, top_k))
