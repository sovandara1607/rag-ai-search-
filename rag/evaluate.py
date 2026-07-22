"""
Small retrieval evaluation harness.

Runs a hand-written set of test queries (each with an expected source document)
through the current VectorStore and reports whether/where the expected doc shows
up in the top-k results. Results and a write-up live in EVALUATION.md.

Run with:
    python rag/evaluate.py
"""

import os

from .ingest import load_documents, build_chunk_records
from .embed_store import VectorStore
from .generate import filter_relevant
from .rerank import rerank_and_aggregate

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
TOP_K = 3
CANDIDATE_K = 15  # wider pool retrieved before reranking narrows back to TOP_K

# "expected_docs" lists every doc title that would be a correct source for the query.
# Most queries have one; a few list two where an old placeholder sample doc and a real
# lecture PDF genuinely cover the same topic (either is an acceptable retrieval).
TEST_QUERIES = [
    {"query": "What happens if I turn in an assignment late?",
     "expected_docs": ["Course Policies"]},
    {"query": "How does BM25 improve on TF-IDF for ranking search results?",
     "expected_docs": ["CS382 Week4"]},
    {"query": "What are the steps to turn raw text into a searchable index, like tokenization and stemming?",
     "expected_docs": ["Week2 SEIR"]},
    {"query": "What are classic information retrieval models like the vector space model?",
     "expected_docs": ["SEIR - Week3 Classic IR"]},
    {"query": "How do you evaluate search quality using precision and recall?",
     "expected_docs": ["SEIR Week 5 - IR Evaluation & Query Processing"]},
    {"query": "What is web crawling and how do search engines discover pages?",
     "expected_docs": ["SEIR Week6 - Web Crawling"]},
    {"query": "How do neural embeddings work for search, and what is a vector database?",
     "expected_docs": ["SEIR Week 8- Neural IR & Vector Databases", "Vector Databases"]},
    {"query": "How does RAG combine retrieval and generation?",
     "expected_docs": ["SEIR Week 9 - RAG Architecture", "Rag Systems"]},
    {"query": "What advanced techniques go beyond basic RAG retrieval?",
     "expected_docs": ["CS382-Week11 Advanced RAG"]},
    {"query": "How does the PageRank algorithm rank web pages using links?",
     "expected_docs": ["SEIR Week12 The PageRank Algorithm"]},
    {"query": "How can search results be biased against certain groups?",
     "expected_docs": ["SEIR Week13 EthicsIR"]},
    {"query": "How do recommender systems suggest items to users?",
     "expected_docs": ["SEIR Week14 Recommender System", "Recommender Systems"]},
    {"query": "What is the cold-start problem in recommendation?",
     "expected_docs": ["SEIR Week14 Recommender System", "Recommender Systems"]},
]


def _retrieve(store, query, use_rerank):
    """Return top-TOP_K (chunk, score) pairs, either straight from FAISS
    cosine similarity or narrowed down via cross-encoder reranking."""
    if not use_rerank:
        return store.query(query, top_k=TOP_K)
    candidates = store.query(query, top_k=CANDIDATE_K)
    floor_passed = filter_relevant(candidates)
    ranked = rerank_and_aggregate(query, floor_passed, TOP_K) if floor_passed else []
    return [(rc.chunk, rc.cosine_score) for rc in ranked]


def run_evaluation(store, use_rerank: bool, label: str):
    hits = 0
    reciprocal_ranks = []
    print(f"\n=== {label} ===")
    print(f"{'Query':<75} {'Rank':<6} {'Top score':<10}")
    print("-" * 95)

    for case in TEST_QUERIES:
        retrieved = _retrieve(store, case["query"], use_rerank)
        doc_titles = [chunk.doc_title for chunk, _ in retrieved]
        rank = next(
            (i + 1 for i, title in enumerate(doc_titles) if title in case["expected_docs"]),
            None,
        )
        top_score = retrieved[0][1] if retrieved else 0.0

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rank_display = str(rank) if rank is not None else "MISS"
        print(f"{case['query']:<75} {rank_display:<6} {top_score:<10.3f}")

    hit_rate = hits / len(TEST_QUERIES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print("-" * 95)
    print(f"Hit rate (top-{TOP_K}): {hit_rate:.0%}  ({hits}/{len(TEST_QUERIES)})")
    print(f"Mean reciprocal rank: {mrr:.2f}")
    return hit_rate, mrr


if __name__ == "__main__":
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)

    baseline = run_evaluation(store, use_rerank=False, label="Baseline (cosine top-k)")
    reranked = run_evaluation(store, use_rerank=True, label="Reranked (cross-encoder)")

    print("\n=== Summary ===")
    print(f"{'':<25} {'Hit rate':<12} {'MRR':<6}")
    print(f"{'Baseline':<25} {baseline[0]:<12.0%} {baseline[1]:<6.2f}")
    print(f"{'Reranked':<25} {reranked[0]:<12.0%} {reranked[1]:<6.2f}")
