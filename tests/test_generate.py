import pytest

from rag.generate import (
    MIN_SIMILARITY,
    NO_MATCH_MESSAGE,
    extractive_answer,
    filter_relevant,
    generate_answer,
    llm_answer,
)
from rag.ingest import Chunk


def _chunk(title="Doc", text="some chunk text", suffix="0"):
    return Chunk(chunk_id=f"{title}::{suffix}", doc_title=title, text=text)


def test_filter_relevant_drops_scores_below_threshold():
    retrieved = [(_chunk(), MIN_SIMILARITY - 0.01)]
    assert filter_relevant(retrieved) == []


def test_filter_relevant_keeps_scores_at_exact_threshold():
    chunk = _chunk()
    retrieved = [(chunk, MIN_SIMILARITY)]
    assert filter_relevant(retrieved) == [(chunk, MIN_SIMILARITY)]


def test_filter_relevant_keeps_scores_above_threshold_and_preserves_order():
    high = _chunk(title="High", suffix="1")
    low = _chunk(title="Low", suffix="2")
    retrieved = [(high, 0.9), (low, 0.01)]
    assert filter_relevant(retrieved) == [(high, 0.9)]


def test_filter_relevant_empty_input_returns_empty():
    assert filter_relevant([]) == []


def test_extractive_answer_no_chunks_returns_no_match_message():
    assert extractive_answer("some query", []) == NO_MATCH_MESSAGE


def test_extractive_answer_cites_document_title_and_score():
    chunk = _chunk(title="Course Policies", text="Late work loses 10% per day.")
    answer = extractive_answer("late policy", [(chunk, 0.42)])
    assert "Course Policies" in answer
    assert "0.42" in answer
    assert "work loses 10% per day." in answer


def test_extractive_answer_picks_most_relevant_sentence_not_whole_chunk():
    chunk = _chunk(
        title="Doc",
        text=(
            "This lecture covers many unrelated topics first. "
            "The cold-start problem happens when a recommender has no data on a new user. "
            "Then it moves on to something else entirely."
        ),
    )
    answer = extractive_answer("what is the cold-start problem?", [(chunk, 0.5)])
    assert "happens when a recommender has no data" in answer
    assert "unrelated topics first" not in answer
    assert "moves on to something else entirely" not in answer


def test_extractive_answer_highlights_matched_query_terms():
    chunk = _chunk(title="Doc", text="BM25 improves ranking over plain TF-IDF scoring.")
    answer = extractive_answer("how does BM25 improve ranking?", [(chunk, 0.6)])
    assert "**BM25**" in answer
    assert "**ranking**" in answer


def test_generate_answer_filters_out_irrelevant_chunks_before_answering():
    relevant = _chunk(title="Relevant", text="relevant text")
    irrelevant = _chunk(title="Irrelevant", text="irrelevant text", suffix="1")
    retrieved = [(relevant, 0.9), (irrelevant, MIN_SIMILARITY - 0.05)]
    answer = generate_answer("q", retrieved, mode="extractive")
    assert "Relevant" in answer
    assert "Irrelevant" not in answer


def test_generate_answer_returns_no_match_message_when_nothing_clears_threshold():
    retrieved = [(_chunk(), 0.01), (_chunk(suffix="1"), 0.02)]
    assert generate_answer("q", retrieved, mode="extractive") == NO_MATCH_MESSAGE


def test_generate_answer_no_match_short_circuits_before_calling_llm(monkeypatch):
    # If nothing is relevant, llm mode must not even attempt an API call.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("llm_answer should not be called when nothing is relevant")

    monkeypatch.setattr("rag.generate.llm_answer", fail_if_called)
    retrieved = [(_chunk(), 0.01)]
    assert generate_answer("q", retrieved, mode="llm") == NO_MATCH_MESSAGE


def test_llm_answer_missing_api_key_falls_back_to_extractive(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    chunk = _chunk(title="Doc", text="chunk text")
    result = llm_answer("q", [(chunk, 0.9)])
    assert "[LLM mode not configured]" in result
    assert "Doc" in result  # falls back to the extractive citation format


def test_llm_answer_empty_retrieved_returns_no_match_message(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert llm_answer("q", []) == NO_MATCH_MESSAGE


def test_llm_answer_success_returns_model_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    class FakeResponse:
        text = "Grounded answer citing Doc."

    class FakeModels:
        def generate_content(self, model, contents):
            assert model == "gemini-2.5-flash"
            assert "chunk text" in contents
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    import google.genai
    monkeypatch.setattr(google.genai, "Client", FakeClient)

    chunk = _chunk(title="Doc", text="chunk text")
    result = llm_answer("q", [(chunk, 0.9)], model="gemini-2.5-flash")
    assert result == "Grounded answer citing Doc."


def test_llm_answer_api_failure_falls_back_to_extractive(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    class FailingModels:
        def generate_content(self, model, contents):
            raise RuntimeError("service unavailable")

    class FailingClient:
        def __init__(self, api_key):
            self.models = FailingModels()

    import google.genai
    monkeypatch.setattr(google.genai, "Client", FailingClient)

    chunk = _chunk(title="Doc", text="chunk text")
    result = llm_answer("q", [(chunk, 0.9)])
    assert "[Gemini call failed" in result
    assert "Doc" in result
