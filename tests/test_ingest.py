from rag.ingest import Chunk, build_chunk_records, chunk_text, derive_title


def _words(n, prefix="word"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_derive_title_capitalizes_lowercase_words():
    assert derive_title("course_policies.txt") == "Course Policies"


def test_derive_title_preserves_existing_casing():
    # Words that aren't all-lowercase (acronyms, mixed case) are left alone.
    assert derive_title("CS382_Week4.pptx.pdf") == "CS382 Week4"


def test_derive_title_strips_pptx_pdf_double_suffix():
    assert derive_title("SEIR_Week13_EthicsIR.pptx.pdf") == "SEIR Week13 EthicsIR"


def test_derive_title_replaces_underscores_with_spaces():
    assert "_" not in derive_title("a_b_c.txt")


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("", chunk_size=80, overlap=20) == []


def test_chunk_text_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\t  ", chunk_size=80, overlap=20) == []


def test_chunk_text_shorter_than_chunk_size_returns_single_chunk():
    text = _words(10)
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert chunks == [text]


def test_chunk_text_covers_all_words_across_chunks():
    text = _words(200)
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    # every word must appear in at least one chunk (no gaps from the stepping logic)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(text.split())


def test_chunk_text_consecutive_chunks_overlap_by_requested_amount():
    text = _words(200)
    chunk_size, overlap = 80, 20
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-overlap:] == second_words[:overlap]


def test_chunk_text_overlap_equal_to_chunk_size_terminates():
    # Regression test: overlap >= chunk_size used to make `start` never
    # advance, hanging the app when the UI sliders allowed this combination.
    text = _words(50)
    chunks = chunk_text(text, chunk_size=10, overlap=10)
    assert len(chunks) > 0
    assert len(chunks) < 1000  # sanity bound, would be unbounded if it looped


def test_chunk_text_overlap_greater_than_chunk_size_terminates():
    text = _words(100)
    chunks = chunk_text(text, chunk_size=40, overlap=50)
    assert len(chunks) > 0
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(text.split())


def test_chunk_text_zero_overlap_produces_disjoint_chunks():
    text = _words(20)
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    expected_second = " ".join(f"word{i}" for i in range(10, 20))
    assert chunks == [_words(10), expected_second]


def test_build_chunk_records_assigns_unique_ids_across_documents():
    docs = [
        {"title": "Doc A", "text": _words(10)},
        {"title": "Doc B", "text": _words(10)},
    ]
    records = build_chunk_records(docs, chunk_size=80, overlap=20)
    ids = [r.chunk_id for r in records]
    assert len(ids) == len(set(ids))
    assert all(isinstance(r, Chunk) for r in records)


def test_build_chunk_records_empty_doc_list_returns_empty():
    assert build_chunk_records([], chunk_size=80, overlap=20) == []


def test_build_chunk_records_skips_chunking_empty_document_text():
    docs = [{"title": "Empty Doc", "text": ""}]
    assert build_chunk_records(docs) == []
