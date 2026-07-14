"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Picks the
  most query-relevant sentence(s) out of each retrieved chunk (rather than
  dumping the whole chunk) and highlights the matched query terms, so you can
  verify retrieval quality before wiring up an LLM.
- "llm": calls Google AI Studio's Gemini API to write a grounded, cited answer
  from the retrieved context. Set the GEMINI_API_KEY environment variable to
  enable it; optionally set GEMINI_MODEL to override the default model.

Retrieval always returns its top_k nearest chunks even when nothing in the
corpus is actually relevant (cosine similarity has no built-in "nothing
matched" signal). `filter_relevant` applies a minimum-similarity floor so
`generate_answer` can tell the two cases apart and avoid fabricating an
answer from off-topic chunks.
"""

import os
import re
from typing import List, Tuple

from .ingest import Chunk

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# Small stopword list used only to pick out query terms worth matching/highlighting
# in extractive snippets - not a general-purpose NLP stopword list.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "but", "with", "at", "by",
    "what", "how", "why", "when", "where", "who", "does", "do", "did",
    "can", "could", "should", "would", "will", "this", "that", "these",
    "those", "it", "its", "as", "from", "about", "i",
}


def _query_terms(query: str) -> List[str]:
    """Extract meaningful words from a query, dropping stopwords/single letters."""
    words = re.findall(r"[a-z0-9']+", query.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _best_snippet(text: str, terms: List[str], max_sentences: int = 2) -> str:
    """Pick the sentence(s) most likely to answer the query out of a chunk,
    instead of returning the whole chunk. Falls back to the first
    `max_sentences` sentences when there's nothing to rank against."""
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences) if sentences else text.strip()

    def score(sentence: str) -> int:
        lowered = sentence.lower()
        return sum(lowered.count(term) for term in terms)

    hits = [(score(s), i) for i, s in enumerate(sentences) if score(s) > 0]
    if not hits:
        return " ".join(sentences[:max_sentences])
    hits.sort(key=lambda item: item[0], reverse=True)
    picked = sorted(i for _, i in hits[:max_sentences])
    return " ".join(sentences[i] for i in picked)


def _highlight(text: str, terms: List[str]) -> str:
    """Bold the query terms that appear in a snippet (markdown `**term**`)."""
    if not terms:
        return text
    unique_terms = sorted(set(terms), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in unique_terms) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)

# Empirically chosen from data/sample_docs (all-MiniLM-L6-v2 + FAISS cosine sim,
# see EVALUATION.md): correct hits scored as low as 0.29, while clearly
# off-topic probe queries (e.g. "chocolate cake recipe") topped out around
# 0.14-0.19. This is a coarse safety net, not a precise classifier, so
# borderline off-topic queries can still slip through — see EVALUATION.md.
MIN_SIMILARITY = 0.25

NO_MATCH_MESSAGE = (
    "I couldn't find relevant information in the indexed documents to answer "
    "that question. Try rephrasing, or check that this topic is covered in "
    "the document collection."
)


def filter_relevant(
    retrieved: List[Tuple[Chunk, float]], min_similarity: float = MIN_SIMILARITY
) -> List[Tuple[Chunk, float]]:
    """Drop retrieved chunks below the relevance floor."""
    return [(chunk, score) for chunk, score in retrieved if score >= min_similarity]


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return NO_MATCH_MESSAGE
    terms = _query_terms(query)
    lines = [f"**Top passages related to:** “{query}”\n"]
    for chunk, score in retrieved:
        snippet = _highlight(_best_snippet(chunk.text, terms), terms)
        lines.append(f"- **{chunk.doc_title}** (score: {score:.2f}) — {snippet}")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]], model: str = None) -> str:
    """Grounded answer via Google AI Studio's Gemini API. Falls back to
    extractive mode if GEMINI_API_KEY isn't set or the API call fails."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return (
            "[LLM mode not configured] Set GEMINI_API_KEY to enable grounded "
            "Gemini answers. Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )
    if not retrieved:
        return NO_MATCH_MESSAGE

    context = "\n\n".join(f"Source: {c.doc_title}\n{c.text}" for c, _ in retrieved)
    prompt = (
        "Answer the question using ONLY the sources below. Cite the source title(s) "
        f"you used.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    from google import genai

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return (
            f"[Gemini call failed: {e}] Falling back to extractive mode:\n\n"
            + extractive_answer(query, retrieved)
        )


def generate_answer(
    query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive", model: str = None
) -> str:
    relevant = filter_relevant(retrieved)
    if not relevant:
        return NO_MATCH_MESSAGE
    if mode == "llm":
        return llm_answer(query, relevant, model=model)
    return extractive_answer(query, relevant)
