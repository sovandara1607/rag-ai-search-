# Retrieval Evaluation

Run with `python -m rag.evaluate` (must be run as a module from the project
root so the relative imports in `rag/` resolve). It sends 13 hand-written test
queries against the full `data/sample_docs/` corpus — 12 real CS382/SEIR
lecture PDFs plus 4 leftover placeholder `.txt` samples from the starter —
checks whether an acceptable source document appears in the top-3 retrieved
chunks, and reports hit rate + mean reciprocal rank (MRR). A few queries list
two acceptable source docs where an old placeholder sample and a real lecture
genuinely cover the same topic.

## Results (embeddings backend: `all-MiniLM-L6-v2` + FAISS, chunk_size=80, overlap=20, 16 docs / 167 chunks)

| Query | Acceptable doc(s) | Rank | Top score |
|---|---|---|---|
| What happens if I turn in an assignment late? | Course Policies | 1 | 0.460 |
| How does BM25 improve on TF-IDF for ranking search results? | CS382 Week4 | 1 | 0.709 |
| What are the steps to turn raw text into a searchable index, like tokenization and stemming? | Week2 SEIR | 1 | 0.564 |
| What are classic information retrieval models like the vector space model? | SEIR - Week3 Classic IR | 1 | 0.622 |
| How do you evaluate search quality using precision and recall? | SEIR Week 5 - IR Evaluation & Query Processing | 1 | 0.643 |
| What is web crawling and how do search engines discover pages? | SEIR Week6 - Web Crawling | 1 | 0.719 |
| How do neural embeddings work for search, and what is a vector database? | SEIR Week 8- Neural IR & Vector Databases, Vector Databases | 1 | 0.598 |
| How does RAG combine retrieval and generation? | SEIR Week 9 - RAG Architecture, Rag Systems | 1 | 0.685 |
| What advanced techniques go beyond basic RAG retrieval? | CS382-Week11 Advanced RAG | **MISS** | 0.607 |
| How does the PageRank algorithm rank web pages using links? | SEIR Week12 The PageRank Algorithm | 1 | 0.740 |
| How can search results be biased against certain groups? | SEIR Week13 EthicsIR | 1 | 0.476 |
| How do recommender systems suggest items to users? | SEIR Week14 Recommender System, Recommender Systems | 1 | 0.745 |
| What is the cold-start problem in recommendation? | SEIR Week14 Recommender System, Recommender Systems | 1 | 0.292 |

**Hit rate (top-3): 92% (12/13)**
**Mean reciprocal rank: 0.92**

## What worked

Once the real CS382/SEIR lecture PDFs were added, retrieval was strong and
confident across nearly every topic — PageRank (0.740), web crawling (0.719),
and BM25 (0.709) all landed a clean rank-1 hit despite the query phrasing
never quoting the slides verbatim. Slide text extracted via `pypdf` is noisy
(titles run together like `"THE MAGICBEHIND THESEARCH BOX"`), but the sentence
embeddings were robust to that noise — semantic meaning survived even when
whitespace didn't.

## What didn't work, and why

The one miss — *"What advanced techniques go beyond basic RAG retrieval?"* —
is caused by the leftover placeholder docs, not the real content. Checking the
full ranking:

```
0.607  Rag Systems (placeholder .txt)         "...generates an answer grounded in the retrieved text..."
0.513  Rag Systems (placeholder .txt)         "Retrieval-Augmented Generation (RAG) RAG systems combine..."
0.496  SEIR Week 9 - RAG Architecture          "...THE OPEN-BOOK TEST ANALOGY..."
0.479  CS382-Week11 Advanced RAG               "...SIMPLE RAG VS. ADVANCED RAG..."
```

`CS382-Week11 Advanced RAG.pdf` actually contains the exact right chunk
(`"SIMPLE RAG VS. ADVANCED RAG"`) — it just ranks 4th, one spot outside
top-3, because the old `rag_systems.txt` sample doc is clean, generic,
textbook-style prose about RAG in general. That kind of writing embeds very
"centrally" in semantic space, so it scores well against almost any
RAG-flavored question even though it's less specific than the real lecture.
The noisier slide text, by contrast, embeds a little further from the query
because of formatting artifacts from PDF extraction.

Takeaway: **leftover placeholder sample docs aren't just harmless clutter —
they can actively outrank more specific real content on generic queries.**
This corpus deliberately keeps them in (per project decision) as a
demonstration of that effect; removing `rag_systems.txt` /
`recommender_systems.txt` / `vector_databases.txt` would likely fix this miss
and is worth trying via a quick `rm` if a cleaner corpus is wanted later.

## Answer quality (generation, `llm` mode via Gemini)

The retrieval table above only measures whether the right document showed up
— it says nothing about whether the *generated answer* is any good, grounded,
or honest when nothing relevant exists. Six representative queries, run
end-to-end through `generate_answer(..., mode="llm")` (Gemini
`gemini-2.5-flash`):

| Query | Behavior |
|---|---|
| How does BM25 improve on TF-IDF for ranking results? | Correct, well-structured answer (term-frequency saturation, length normalization, smoothed IDF), cited `CS382 Week4` throughout. |
| What advanced techniques go beyond basic RAG retrieval? | Honestly said the sources didn't cover it, rather than guessing — because the retrieval miss above means the real "Advanced RAG" PDF chunk wasn't in the top-3 context it was given. |
| What is the cold-start problem in recommendation? | Correct, cited `SEIR Week14 Recommender System`, and low top score (0.29) still cleared the 0.25 relevance floor. |
| How can search results be biased against certain groups? | Synthesized three distinct points (disparate impact, personalization/"echo chamber", relevance being user-dependent) across two source documents with citations for each. |
| What is the best recipe for chocolate cake? | Correctly refused — top score 0.14, below the relevance floor, so the LLM was never called and the "nothing relevant found" message was returned instead. |
| Who won the 2022 soccer World Cup? | Same graceful refusal as above (top score 0.19). |

## Successes

- Grounded citations are accurate: every generated answer names the specific
  source document(s) it drew from, and spot-checking those citations against
  the actual chunk text confirms the claims are actually supported by it.
- The relevance floor (`MIN_SIMILARITY = 0.25` in `rag/generate.py`) correctly
  rejects clearly off-topic queries (chocolate cake, World Cup) without ever
  calling the LLM, satisfying the "don't hallucinate when nothing's relevant"
  requirement and saving an API call in the process.
- When retrieval *does* surface the right chunks, the LLM does not
  overreach beyond them — see the "advanced RAG techniques" answer, which
  stated the limitation instead of fabricating advanced-RAG content it hadn't
  actually been given.

## Limitations

- **Generation quality is bottlenecked by retrieval.** The one retrieval miss
  (advanced RAG techniques) directly caused an incomplete answer even though
  generation itself behaved correctly — the LLM can only be grounded in
  whatever the retriever hands it.
- **The relevance floor is a coarse heuristic, not a learned classifier.** It's
  a single global cosine-similarity cutoff picked by inspecting score
  distributions on this corpus (see "What didn't work" above), not something
  that adapts to query type or corpus. A borderline off-topic query that
  happens to share vocabulary with the corpus (e.g. a question about
  "ranking" in a non-search-engine sense) could still slip past the 0.25
  floor and get an answer synthesized from marginally-related chunks.
- **PDF slide extraction is noisy** (see "What worked" above — mashed-together
  title text), which occasionally shows up verbatim in generated answers
  (e.g. quoted fragments with missing spaces) even though the underlying
  claim is correct.
- **No multi-hop reasoning.** Each answer is generated from a single
  retrieval pass; a question that requires combining facts from two unrelated
  parts of the corpus that don't share vocabulary (so both wouldn't be
  retrieved together) would not be answered completely.
