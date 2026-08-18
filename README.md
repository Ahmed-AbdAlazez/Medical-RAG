# Medical RAG Assistant — ADHD Clinical Guidance

A Retrieval-Augmented Generation (RAG) system built for the **AI Clinical Decision Support Lite Hackathon**. It answers questions strictly from two source PDFs — a NICE clinical guideline on ADHD and a parenting guide — with grounded citations, and refuses to answer anything outside that scope.

## Source documents

| File | Content |
|---|---|
| `Data/ADHD.pdf` | NICE guideline NG87 — ADHD diagnosis and management (61 pages) |
| `Data/parenting2015.pdf` | Practical parenting guidance for children with ADHD |

## Pipeline

```
PDFs → Parsing → Chunking → Embedding → Hybrid Index → Retrieval → Groq LLM → Answer + Citations
```

### 1. Parsing (`ingest.py`)
- `PyPDFLoader` extracts text per page via `pypdf`
- Cleans watermarks, footers, and repeated boilerplate (`clean_text`)
- Tags each page with `document_name`, 1-indexed `page_number`, a `citation` key (`ADHD.pdf:p46`), and an inferred `section` heading

### 2. Chunking
- Section-aware recursive splitter (`RecursiveCharacterTextSplitter`) — tries paragraph, line, sentence, semicolon, colon, then word/character boundaries
- Default: **400-token chunks, 80-token overlap** (`config.CHUNK_SIZE` / `CHUNK_OVERLAP`)
- Alternate setting tested: 600/100 (see evaluation below)

### 3. Embeddings (`embedding_models.py`)
- **`JinaSmallEmbedding`** — calls the Jina API (`jina-embeddings-v2-small-en`) when `JINA_API_KEY` is set
- **`TfidfEmbedding`** — local, corpus-aware TF-IDF fallback so the system runs fully offline if no key is present
- Falls back automatically and transparently on API failure

### 4. Retrieval (`ingest.py: SimpleVectorIndex`)
- **Hybrid search**: `0.65 × semantic similarity + 0.35 × keyword (TF-IDF) score`
- Returns top-K chunks (`config.TOP_K = 3`) with `semantic_score` and `keyword_score` recorded per result

### 5. Answer generation
- Retrieved chunks are assembled into a context block and sent to **Groq** (`openai/gpt-oss-20b`) with a strict system prompt:
  - Answer only from the provided context
  - No outside knowledge, no guessing
  - Explicit refusal message if the context doesn't contain the answer
- Citations and the retrieved chunks are returned alongside the answer

## Interfaces

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat UI — cached index, chat history, expandable citations/chunks per answer |
| `compare_embedding_models.py` | Terminal chat loop for interactively testing the RAG answers |
| `evaluate_retrieval_metrics.py` | Batch precision/recall@3 evaluation across chunk sizes and embedding models |
| `Day1_Task1_Document_Ingestion.ipynb` | Step-by-step walkthrough notebook: parsing → chunking → embedding → retrieval |

## Setup

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
JINA_API_KEY=your_jina_key   # optional — falls back to TF-IDF if omitted
```

Run the app:

```powershell
python -m streamlit run app.py
```

## Evaluation results

Offline retrieval metrics (TF-IDF fallback, no API keys) across chunk settings:

| Chunk setting | Model | Chunks | Precision@3 | Recall@3 |
|---|---|---:|---:|---:|
| 400 / 80 | JinaSmallEmbedding (TF-IDF fallback) | 110 | 0.333 | 1.000 |
| 400 / 80 | TfidfEmbedding | 110 | 0.333 | 1.000 |
| 600 / 100 | JinaSmallEmbedding (TF-IDF fallback) | 70 | 0.333 | 1.000 |
| 600 / 100 | TfidfEmbedding | 70 | 0.333 | 1.000 |

Both settings retrieve the correct citation in the top 3 for all test questions (recall = 1.0); precision is capped at 0.333 because only 1 of the 3 retrieved chunks per question is the labeled ground truth.

### Manual Streamlit testing (Groq + Jina live)

| Question | Expected citation | Result |
|---|---|---|
| Committee recommendations for responsible ADHD medication use | `ADHD.pdf:p46` | ✅ Top result, score 0.933 |
| Discipline methods parents should learn | `parenting2015.pdf:p3` | ✅ Top result, score 0.917 |
| First-line medication for children 5+ | `ADHD.pdf:p26` (rec. 1.7.7) | ✅ Top result, score 0.924 |
| Height/weight monitoring frequency | `ADHD.pdf:p32`–`p33` (rec. 1.8.5) | ✅ Top 2 results, scores 0.923 / 0.856 |

All four spot-checks returned the correct source page as the top-ranked (or top-2) hybrid search result, with answers grounded in the retrieved text and no hallucinated content.

## Known limitations

- TF-IDF fallback vocabulary is capped (`TFIDF_MAX_FEATURES = 512`), which can miss rare terms if `JINA_API_KEY` isn't set
- Precision@3 is mechanically low by design — only one labeled ground-truth citation per question, even when multiple retrieved chunks are contextually useful
- No re-ranking step after hybrid search; relies entirely on the weighted semantic+keyword score
