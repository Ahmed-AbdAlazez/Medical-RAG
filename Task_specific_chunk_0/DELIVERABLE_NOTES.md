# Day 1 Document Ingestion Notes

## What changed

The pipeline now ingests both PDFs in `Data`: `ADHD.pdf` and `parenting2015.pdf`. Parsing normalizes document name, 1-indexed page number, citation key, and section metadata for every page and chunk.

## Chunking strategy

The code uses a section-aware recursive splitter with paragraph, line, sentence, semicolon, colon, word, and final character boundaries. `Task_specific_chunk_0` uses 400-token chunks with 80-token overlap. `Task_specific_chunk_1` uses 600-token chunks with 100-token overlap.

## Embedding model

`HashingEmbedding` was removed. The main embedding wrapper is now `JinaSmallEmbedding`, configured for `jina-embeddings-v2-small-en`. If `JINA_API_KEY` is missing, it falls back to the local `TfidfEmbedding` so the notebook remains runnable offline.

## Retrieval

Retrieval uses hybrid search: semantic similarity plus keyword matching. Results show citation metadata as `document_name:ppage`, for example `ADHD.pdf:p46` and `parenting2015.pdf:p1`.
