"""
RAG-Based AI Search System — interface, styled as a library card-catalog
desk for the CS382/SEIR course archive.

Run with:
    streamlit run app.py

Document loading, embedding-based retrieval (sentence-transformers + FAISS),
and an extractive answer — wired into a Streamlit UI with file upload,
adjustable chunking, term highlighting, and latency display. Build on this by
upgrading `rag/generate.py`'s `llm_answer` (see its TODO) once you have an
LLM API key.
"""

import html
import re
import time

import streamlit as st
from dotenv import load_dotenv

from rag.ingest import load_documents, build_chunk_records, derive_title, read_uploaded_file
from rag.embed_store import VectorStore
from rag.generate import generate_answer, filter_relevant, MIN_SIMILARITY, DEFAULT_GEMINI_MODEL
from rag.rerank import rerank_and_aggregate

load_dotenv()

GEMINI_MODEL_CHOICES = ["gemini-flash-latest", "gemini-3-flash-preview", "gemini-flash-lite-latest", "Custom..."]

DATA_FOLDER = "data/sample_docs"

st.set_page_config(page_title="CS382 / SEIR course archive", page_icon=":material/search:", layout="wide")

# ---------------------------------------------------------------------------
# Signature styling: a small, targeted CSS layer for the elements native
# theming can't express — the stamp badge, source index cards, and relevance
# meter. Colors/fonts/radii themselves live in .streamlit/config.toml.
# ---------------------------------------------------------------------------
st.html("""
<style>
  h1, h2, h3 { letter-spacing: -0.01em; font-weight: 700; }

  /* Faint dot-grid on near-black, no gradients/blur — minimal & flat. */
  [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
    background-color: #0A0A0A;
    background-image: radial-gradient(rgba(255,255,255,0.09) 1px, transparent 1px);
    background-size: 22px 22px;
  }
  [data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid #1F1F1F;
  }

  .eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #71717A;
    margin: 0 0 0.35rem 0;
  }

  .hero-tagline {
    color: #9CA3AF;
    max-width: 46em;
    margin-top: -0.4rem;
  }

  /* Query slip — flat rounded-rectangle prompt bar with a chip row
     underneath for the active search settings. */
  div[class*="st-key-query-slip"] {
    background: #111113;
    border: 1px solid #262626;
    border-radius: 20px;
    padding: 0.55rem 0.6rem 0.55rem 1.4rem;
    margin-bottom: 1.1rem;
    transition: border-color 160ms ease;
  }
  div[class*="st-key-query-slip"]:focus-within {
    border-color: #52525B;
  }
  div[class*="st-key-query-slip"] label p { color: #FAFAFA !important; }
  div[class*="st-key-query-slip"] div[data-testid="stTextInputRootElement"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
  }
  div[class*="st-key-query-slip"] input {
    font-size: 1rem !important;
    color: #FAFAFA !important;
    background: transparent !important;
  }
  div[class*="st-key-query-slip"] input::placeholder { color: #52525B; opacity: 1; }
  div[class*="st-key-query-slip"] div[data-testid="stButton"] button {
    border-radius: 12px !important;
    height: 2.75rem;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
  }
  div[class*="st-key-query-slip"] button[data-testid="stBaseButton-primary"] {
    background: #FAFAFA !important;
    border: 1px solid #FAFAFA !important;
    color: #0A0A0A !important;
  }
  div[class*="st-key-query-slip"] button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border-color: #262626 !important;
    color: #A1A1AA !important;
  }

  /* Streamlit stacks st.columns below a 640px container width (each column
     gets min-width: calc(100% - Npx)). Left un-overridden, the input and
     send button drop onto separate rows inside this pill-shaped bar and the
     999px border-radius mangles the shape. Force a single row at any width
     instead, letting the input shrink and the button stay compact. */
  div[class*="st-key-query-slip"] div[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
  }
  div[class*="st-key-query-slip"] div[data-testid="stColumn"] {
    min-width: 0 !important;
    width: auto !important;
  }
  div[class*="st-key-query-slip"] div[data-testid="stColumn"]:first-of-type {
    flex: 1 1 auto !important;
  }
  div[class*="st-key-query-slip"] div[data-testid="stColumn"]:last-of-type {
    flex: 0 0 auto !important;
  }

  /* Answer-mode segmented control — restyle into a compact pill toggle
     that matches the chip row instead of Streamlit's boxy default. */
  div[class*="st-key-mode-select"] [data-testid="stButtonGroup"] {
    padding-left: 1.6rem;
  }
  div[class*="st-key-mode-select"] div[role="radiogroup"] {
    display: inline-flex;
    background: #0A0A0A;
    border: 1px solid #262626;
    border-radius: 999px;
    padding: 3px;
    gap: 2px;
  }
  div[class*="st-key-mode-select"] button[data-variant="segmented_control"] {
    border: none !important;
    border-radius: 999px !important;
    background: transparent !important;
    color: #71717A !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    padding: 0.32rem 0.9rem !important;
    min-height: 0 !important;
    transition: background 140ms ease, color 140ms ease;
  }
  div[class*="st-key-mode-select"] button[data-variant="segmented_control"]:hover {
    color: #D4D4D8 !important;
  }
  div[class*="st-key-mode-select"] button[data-variant="segmented_control"][aria-checked="true"] {
    background: #FAFAFA !important;
    color: #0A0A0A !important;
  }
  div[class*="st-key-mode-select"] button[data-variant="segmented_control"][aria-checked="true"]:hover {
    color: #0A0A0A !important;
  }

  /* Settings chip row under the prompt bar */
  .prompt-chips {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    padding: 0 0.4rem 1rem 1.6rem;
    margin-top: -0.6rem;
  }
  .prompt-chip {
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #A1A1AA;
    background: #111113;
    border: 1px solid #262626;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
  }
  .prompt-chip-hint {
    font-size: 0.72rem;
    color: #52525B;
  }

  /* Answer card */
  div[class*="st-key-answer-card"] {
    position: relative;
    background: #111113;
    border: 1px solid #262626;
    border-radius: 16px;
    padding: 1.4rem 1.6rem 1.2rem 1.6rem;
    margin-bottom: 1.8rem;
    overflow: visible;
  }
  div[class*="st-key-answer-card"] p,
  div[class*="st-key-answer-card"] li { color: #D4D4D8 !important; }
  div[class*="st-key-answer-card"] h3 { color: #FAFAFA !important; margin-top: 0; }

  .stamp-badge {
    position: absolute;
    top: -14px;
    right: 24px;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    letter-spacing: 0.03em;
    color: #D4D4D8;
    background: #18181B;
    border: 1px solid #262626;
    border-radius: 999px;
    padding: 5px 12px 5px 10px;
  }
  .stamp-badge .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22D3EE;
    box-shadow: 0 0 6px rgba(34,211,238,0.8);
  }

  /* Source index cards */
  div[class*="st-key-src-"] {
    position: relative;
    background: #111113;
    border: 1px solid #262626;
    border-radius: 14px;
    padding: 0.9rem 1.1rem 0.3rem 1.1rem;
    margin-bottom: 0.9rem;
    transition: border-color 160ms ease, transform 160ms ease;
  }
  div[class*="st-key-src-"]:hover {
    border-color: #3F3F46;
    transform: translateY(-2px);
  }
  div[class*="st-key-src-"] [data-testid="stExpander"] {
    background: transparent;
    border: none;
  }
  div[class*="st-key-src-"] summary,
  div[class*="st-key-src-"] summary p { color: #E4E4E7 !important; }
  div[class*="st-key-src-"] [data-testid="stExpanderDetails"] p { color: #D4D4D8 !important; }

  .call-number {
    font-size: 0.82rem;
    color: #FAFAFA;
    margin-bottom: 0.5rem;
  }
  .call-number .doc { font-weight: 600; }
  .call-number .chunk { color: #71717A; }

  .rel-meter { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.7rem; }
  .rel-track {
    position: relative;
    flex: 1 1 auto;
    height: 4px;
    background: #262626;
    border-radius: 2px;
  }
  .rel-fill {
    position: absolute; left: 0; top: 0; height: 100%;
    background: #22D3EE;
    border-radius: 2px;
  }
  .rel-tick {
    position: absolute; top: -3px; width: 1px; height: 10px;
    background: #52525B;
  }
  .rel-tick-label {
    position: absolute; top: 8px; transform: translateX(-50%);
    font-size: 0.62rem; color: #71717A;
    white-space: nowrap;
  }
  .rel-score {
    font-size: 0.85rem;
    font-weight: 600;
    color: #22D3EE;
    min-width: 3.2em;
    text-align: right;
  }

  .rerank-badge {
    display: inline-block;
    font-size: 0.68rem;
    letter-spacing: 0.03em;
    color: #A78BFA;
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 999px;
    padding: 0.12rem 0.55rem;
    margin-bottom: 0.6rem;
  }

  mark {
    background: rgba(34,211,238,0.16);
    color: #67E8F9;
    border-radius: 3px;
    padding: 0 3px;
    text-decoration: none;
    font-weight: 600;
  }

  :focus-visible {
    outline: 2px solid #22D3EE !important;
    outline-offset: 2px;
    box-shadow: none !important;
  }
</style>
""")


@st.cache_resource(show_spinner="Loading and indexing documents...")
def build_store(chunk_size: int, overlap: int, extra_docs: list):
    docs = load_documents(DATA_FOLDER) + list(extra_docs)
    chunks = build_chunk_records(docs, chunk_size=chunk_size, overlap=overlap)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


def highlight_terms(text: str, query: str) -> str:
    """HTML-escape chunk text and wrap query terms in <mark> for display."""
    escaped = html.escape(text)
    terms = sorted({t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}, key=len, reverse=True)
    if not terms:
        return escaped
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped)


def relevance_meter_html(score: float, floor: float) -> str:
    """A 0-1 axis showing where a chunk's similarity score sits relative to
    the app's relevance floor — makes the actual cutoff mechanism visible
    instead of just printing a bare number."""
    fill_pct = max(0.0, min(score, 1.0)) * 100
    floor_pct = floor * 100
    return f"""
    <div class="rel-meter">
      <div class="rel-track">
        <div class="rel-fill" style="width:{fill_pct:.1f}%;"></div>
        <div class="rel-tick" style="left:{floor_pct:.1f}%;"></div>
        <div class="rel-tick-label" style="left:{floor_pct:.1f}%;">floor {floor:.2f}</div>
      </div>
      <div class="rel-score">{score:.2f}</div>
    </div>
    """


def passage_label(chunk_id: str, total_in_doc: int) -> str:
    """Turn a chunk_id's trailing index (`::3`) or merged range (`::3-4`,
    from aggregate_adjacent) into a 1-indexed display label."""
    suffix = chunk_id.rsplit("::", 1)[-1]
    if "-" in suffix:
        start, end = (int(n) for n in suffix.split("-", 1))
        return f"passages {start + 1}–{end + 1} of {total_in_doc}"
    return f"passage {int(suffix) + 1} of {total_in_doc}"


def call_number_html(doc_title: str, label: str) -> str:
    return (
        f'<div class="call-number">'
        f'<span class="doc">{html.escape(doc_title)}</span>'
        f'<span class="chunk"> &nbsp;·&nbsp; {label}</span>'
        f"</div>"
    )


def rerank_badge_html(score: float) -> str:
    return f'<div class="rerank-badge">reranked &middot; cross-encoder {score:.2f}</div>'


with st.sidebar:
    #st.markdown('<p class="eyebrow">Drawer 01 · retrieval &amp; generation</p>', unsafe_allow_html=True)
    st.header("Catalog settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    st.caption(f"Retrieving top **{top_k}** passages per search.")
    use_rerank = st.checkbox(
        "Rerank with cross-encoder",
        value=True,
        help="Retrieves a wider candidate pool and reorders it with a cross-encoder "
        "(ms-marco-MiniLM-L-6-v2) before generation — usually more accurate than cosine "
        "similarity alone, at the cost of a bit of latency.",
    )
    st.caption("Answer mode and model are chosen in the search box below.")

    with st.expander("Chunking settings", icon=":material/tune:"):
        chunk_size = st.slider("Chunk size (words)", min_value=40, max_value=200, value=80, step=10)
        overlap = st.slider("Chunk overlap (words)", min_value=0, max_value=50, value=20, step=5)

    uploaded_files = st.file_uploader(
        "Add your own .txt or .pdf documents", type=["txt", "pdf"], accept_multiple_files=True
    )
    extra_docs = []
    for uploaded in uploaded_files or []:
        text = read_uploaded_file(uploaded)
        if text:
            extra_docs.append({"title": derive_title(uploaded.name), "text": text})

store, docs, chunks = build_store(chunk_size, overlap, extra_docs)

chunks_per_doc = {}
for c in chunks:
    chunks_per_doc[c.doc_title] = chunks_per_doc.get(c.doc_title, 0) + 1

with st.sidebar:
    st.caption(f"Indexed **{len(docs)}** documents → **{len(chunks)}** chunks")
    with st.expander("Documents in this index", icon=":material/folder:"):
        for d in docs:
            st.write(f"- {d['title']}")

st.markdown('<p class="eyebrow">RAG search</p>', unsafe_allow_html=True)
st.title("CS382 / SEIR course archive")
st.markdown(
    '<p class="hero-tagline">A card catalog for the lecture notes, not a chatbot. '
    "Ask a question and it searches the indexed corpus below — every answer is grounded "
    "in retrieved passages, cited and scored, or it tells you plainly that nothing matched.</p>",
    unsafe_allow_html=True,
)

with st.container(key="query-slip"):
    input_col, button_col = st.columns([11, 2], vertical_alignment="center", gap="small")
    with input_col:
        query = st.text_input(
            "Your question",
            placeholder="e.g. How does content-based filtering rank items?",
            label_visibility="collapsed",
        )
    has_query = bool(query and query.strip())
    with button_col:
        search_clicked = st.button(
            "Ask" if has_query else "Search",
            icon=":material/arrow_upward:" if has_query else ":material/search:",
            type="primary" if has_query else "secondary",
            key="send-btn",
            width="stretch",
        )
    mode_col, model_col, hint_col = st.columns([2, 3, 4], vertical_alignment="center", gap="small")
    with mode_col:
        mode_label = st.segmented_control(
            "Answer mode",
            options=["Extractive", "LLM"],
            default="Extractive",
            required=True,
            label_visibility="collapsed",
            key="mode-select",
            help="Extractive works with no setup. LLM mode (Gemini) needs GEMINI_API_KEY set.",
        )
    mode = "llm" if mode_label == "LLM" else "extractive"

    gemini_model = DEFAULT_GEMINI_MODEL
    if mode == "llm":
        with model_col:
            model_choice = st.selectbox(
                "Gemini model",
                GEMINI_MODEL_CHOICES,
                index=0,
                label_visibility="collapsed",
                key="gemini-model-select",
            )
        if model_choice == "Custom...":
            with hint_col:
                gemini_model = st.text_input(
                    "Custom model name",
                    value=DEFAULT_GEMINI_MODEL,
                    label_visibility="collapsed",
                    key="custom-model",
                )
        else:
            gemini_model = model_choice

    st.markdown(
        '<div class="prompt-chips">'
        '<span class="prompt-chip-hint">Press Enter ↵ to search</span>'
        "</div>",
        unsafe_allow_html=True,
    )

if search_clicked and query.strip():
    start = time.perf_counter()
    rerank_scores = {}
    if use_rerank:
        candidate_k = min(max(top_k * 4, 15), len(chunks))
        candidates = store.query(query, top_k=candidate_k)
        floor_passed = filter_relevant(candidates)
        ranked = rerank_and_aggregate(query, floor_passed, top_k) if floor_passed else []
        relevant = [(rc.chunk, rc.cosine_score) for rc in ranked]
        rerank_scores = {rc.chunk.chunk_id: rc.rerank_score for rc in ranked}
        answer = generate_answer(query, relevant, mode=mode, model=gemini_model)
    else:
        retrieved = store.query(query, top_k=top_k)
        relevant = filter_relevant(retrieved)
        answer = generate_answer(query, retrieved, mode=mode, model=gemini_model)
    elapsed_ms = (time.perf_counter() - start) * 1000

    with st.container(key="answer-card"):
        st.markdown(
            f'<div class="stamp-badge"><span class="status-dot"></span>Answered · {elapsed_ms:.0f} ms</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Answer")
        st.write(answer)

    st.header(f"Sources ({len(relevant)})")
    if not relevant:
        st.info(f"No retrieved chunk cleared the relevance floor (similarity ≥ {MIN_SIMILARITY:.2f}).")
    else:
        for i, (chunk, score) in enumerate(relevant):
            total_in_doc = chunks_per_doc.get(chunk.doc_title, 1)
            with st.container(key=f"src-{i}"):
                st.markdown(call_number_html(chunk.doc_title, passage_label(chunk.chunk_id, total_in_doc)), unsafe_allow_html=True)
                if chunk.chunk_id in rerank_scores:
                    st.markdown(rerank_badge_html(rerank_scores[chunk.chunk_id]), unsafe_allow_html=True)
                st.markdown(relevance_meter_html(score, MIN_SIMILARITY), unsafe_allow_html=True)
                with st.expander("Read excerpt", icon=":material/description:"):
                    st.markdown(highlight_terms(chunk.text, query), unsafe_allow_html=True)
elif search_clicked:
    st.warning("Type a question first.")
