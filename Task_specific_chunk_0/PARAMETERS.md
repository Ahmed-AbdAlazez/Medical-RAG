# Parameters

- Source PDFs: `ADHD.pdf`, `parenting2015.pdf`
- Chunk size: `400` tokens for chunk_0, `600` tokens for chunk_1
- Chunk overlap: `80` tokens for chunk_0, `100` tokens for chunk_1
- Main embedding model: `JinaSmallEmbedding`
- Jina model name: `jina-embeddings-v2-small-en`
- Fallback model if `JINA_API_KEY` is missing: `TfidfEmbedding`
- Alternative embedding model tested: `TfidfEmbedding`
- Search method: `Hybrid Search`
- Hybrid weights: `0.65` semantic, `0.35` keyword
