import json
from pathlib import Path


NOTEBOOK = Path("Day1_Task1_Document_Ingestion.ipynb")


def source(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = nb["cells"]

cells[0]["source"] = source(
    """
# Day 1 - Document Ingestion
### AI Clinical Decision Support Lite Hackathon - Plan B

This notebook walks through the full Day 1 pipeline step by step: parsing a real clinical
guideline PDF, choosing a chunking strategy, generating embeddings, and building a
queryable vector index.

**By the end of this notebook you will be able to:**
1. Explain why grounding - not raw model memory - matters in clinical AI
2. Parse a PDF and inspect its extracted structure
3. Compare fixed-size vs. section-aware chunking on the same document
4. Generate an embedding and explain what the resulting vector represents
5. Build a persisted vector index and run a real query against it

> **Data source:** this notebook uses `Data/ADHD.pdf`, the NICE guideline
> "Attention deficit hyperactivity disorder: diagnosis and management (NG87)".
"""
)

cells[2]["source"] = source(
    """
import sys, os
sys.path.append(os.path.abspath("."))

import config
from pathlib import Path

print("Data directory:", config.DATA_DIR)
print("Chunk size (tokens):", config.CHUNK_SIZE)
print("Chunk overlap (tokens):", config.CHUNK_OVERLAP)
print("Hybrid semantic weight:", config.HYBRID_SEMANTIC_WEIGHT)
print("Hybrid keyword weight:", config.HYBRID_KEYWORD_WEIGHT)
print("PDFs found:", [p.name for p in config.DATA_DIR.glob("*.pdf")])
"""
)

cells[3]["source"] = source(
    """
assert config.DATA_DIR.exists(), f"Data directory not found: {config.DATA_DIR}"
assert list(config.DATA_DIR.glob("*.pdf")), "No PDF files found for ingestion."
"""
)

cells[6]["source"] = source(
    """
from ingest import PyPDFLoader

pdf_path = list(config.DATA_DIR.glob("*.pdf"))[0]
print(f"Loading: {pdf_path.name}\\n")

loader = PyPDFLoader(str(pdf_path))
raw_pages = loader.load()

print(f"Loaded {len(raw_pages)} pages.\\n")
print("--- Page 3 (index 2) raw metadata, as PyPDFLoader gives it to us ---")
print(raw_pages[2].metadata)
print("\\n--- Page 3 (index 2) first 400 characters ---")
print(raw_pages[2].page_content[:400])
"""
)

cells[9]["source"] = source(
    """
### Checkpoint 1

Look at the printed text above. Answer for yourself before moving on:

- Are section headings such as "1.1 Service organisation and training" visible as recognizable text?
- Are there any obvious parsing artifacts (broken words, merged columns, stray characters)?

If parsing looks clean here, section-aware chunking (Step 2) will work well. If it looks
messy, no chunking strategy will fully save you - the fix belongs upstream, in parsing.
"""
)

cells[11]["source"] = source(
    """
#!-------------------------------!
# NOTE: You can use whatever text splitter you prefer, e.g. NLTK or spaCy.
#!-------------------------------!

from ingest import RecursiveCharacterTextSplitter

# --- Naive fixed-size splitter: no regard for sentence/paragraph boundaries ---
naive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # characters, not tokens - deliberately crude
    chunk_overlap=0,
    separators=[""],       # forces raw character-count splitting
)
naive_chunks = naive_splitter.split_documents(pages)

# --- Section-aware splitter: same one used in ingest.py ---
aware_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE * 4,      # ~4 chars/token estimate
    chunk_overlap=config.CHUNK_OVERLAP * 4,
    separators=["\\n\\n", "\\n", ". ", " ", ""],
)
aware_chunks = aware_splitter.split_documents(pages)

print(f"Naive fixed-size chunker:   {len(naive_chunks)} chunks")
print(f"Section-aware chunker:     {len(aware_chunks)} chunks")
"""
)

cells[17]["source"] = source(
    """
from ingest import cosine_similarity, get_embedding_function

embed_fn = get_embedding_function()

texts = [
    "parents can create home and school environments for a child with ADHD",
    "families can support a child with attention deficit hyperactivity disorder at home and school",
    "recommended screening interval for breast cancer",
]

vectors = embed_fn.embed_documents(texts)
print(f"Each embedding is a vector of length {len(vectors[0])}\\n")

sim_related = cosine_similarity(vectors[0], vectors[1])
sim_unrelated = cosine_similarity(vectors[0], vectors[2])

print(f"Similarity - same meaning, different words:  {sim_related:.3f}")
print(f"Similarity - genuinely different topics:      {sim_unrelated:.3f}")
"""
)

cells[19]["source"] = source(
    """
## 6. Step 5 - Build the Vector Index

Now we embed every chunk and store it in a local persisted vector index, using the exact
same `build_index()` function from `ingest.py`. This keeps the notebook runnable in
restricted environments while preserving the same retrieval workflow.
"""
)

cells[22]["source"] = source(
    """
question = "How can parents create better home and school environments for a child with ADHD?"

results = vectordb.hybrid_search_with_relevance_scores(
    question,
    k=config.TOP_K,
    semantic_weight=config.HYBRID_SEMANTIC_WEIGHT,
    keyword_weight=config.HYBRID_KEYWORD_WEIGHT,
)

print(f"Question: {question}\\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] hybrid={score:.3f}  semantic={doc.metadata.get('semantic_score')}  "
          f"keyword={doc.metadata.get('keyword_score')}  {doc.metadata.get('document_name')}, "
          f"page {doc.metadata.get('page_number')}, section: {doc.metadata.get('section')}")
    print(f"    \\"{doc.page_content[:220].strip()}...\\"\\n")
"""
)

NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
